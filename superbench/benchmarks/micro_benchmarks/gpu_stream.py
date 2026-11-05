# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPU STREAM bandwidth with inline launch overrides and a fixed fallback sweep."""

import json
import math
import os
import re
import shlex

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


_OPERATIONS = ['copy', 'scale', 'add', 'triad']
_BLOCKS = [128, 256, 512, 1024]
_UNROLLS = [1, 2, 3, 4, 6, 8]
_DEFAULT_SIZE = 8 * 1024**3
_TAG = re.compile(r'^STREAM_(COPY|SCALE|ADD|TRIAD)_(float|double)$')


class GpuStreamBenchmark(MicroBenchmarkWithInvoke):
    """Report one bandwidth per operation and datatype."""

    def __init__(self, name, parameters=''):
        """Initialize the native benchmark."""
        super().__init__(name, parameters)
        self._bin_name = 'gpu_stream'
        self._command_targets = []

    def add_parser_arguments(self):
        """Retain the normal benchmark arguments; search dimensions are fixed."""
        super().add_parser_arguments()
        self._parser.add_argument('--kernel_configs', type=str, default=None,
                                  help='JSON mapping of workload tags to block_size/unroll. Omit to autotune.')
        self._parser.add_argument('--size', type=int, default=_DEFAULT_SIZE,
                                  help='Bytes per array, independent of launch overrides (default: 8 GiB).')
        self._parser.add_argument('--num_warm_up', type=int, default=20)
        self._parser.add_argument('--num_loops', type=int, default=100)
        self._parser.add_argument('--data_type', choices=['float', 'double'], default='double',
                                  help='Element type; use one benchmark entry per data type.')
        self._parser.add_argument('--check_data', action='store_true',
                                  help='Validate output. Requires 6x buffer size in host memory; use small sizes.')

    @staticmethod
    def _check_config(tag, config):
        """Check only the parameters needed to launch a workload."""
        match = _TAG.fullmatch(tag)
        if match is None or not isinstance(config, dict):
            raise ValueError('Invalid workload configuration key: {}'.format(tag))
        if set(config) != {'block_size', 'unroll'} or any(type(value) is not int for value in config.values()):
            raise ValueError('{} needs integer block_size and unroll'.format(tag))
        if config['block_size'] not in _BLOCKS or config['unroll'] not in _UNROLLS:
            raise ValueError('Unsupported block size/unroll for {}'.format(tag))

    def _kernel_configs(self):
        """Load explicit workload overrides for the selected data type only."""
        if self._args.kernel_configs is None:
            return {}
        configs = json.loads(self._args.kernel_configs)
        if not isinstance(configs, dict):
            raise ValueError('Expected a workload-tag to configuration mapping')
        for tag, config in configs.items():
            self._check_config(tag, config)
            if not tag.endswith('_' + self._args.data_type):
                raise ValueError('{} does not match data_type {}'.format(tag, self._args.data_type))
        return configs

    def _command(self, dtype, operation, size, config=None):
        """Use singleton native parameters for overrides, or the native fixed sweep."""
        parts = [
            shlex.quote(self.__bin_path), '--size', str(size), '--data_type', dtype, '--operation', operation,
            '--num_warm_up', str(self._args.num_warm_up), '--num_loops', str(self._args.num_loops),
        ]
        if config:
            parts.extend(['--block_size', str(config['block_size']), '--unroll', str(config['unroll'])])
        if self._args.check_data:
            parts.append('--check_data')
        return ' '.join(parts)

    def _preprocess(self):
        """Resolve size once, independently of workload launch overrides."""
        if not super()._preprocess():
            return False
        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)
        self._commands, self._command_targets = [], []
        try:
            if (self._args.num_warm_up < 0 or self._args.num_loops <= 0 or
                    self._args.num_warm_up + self._args.num_loops >= 2**64):
                raise ValueError('Invalid warm-up/loop counts')
            width = 4 if self._args.data_type == 'float' else 8
            if not 0 < self._args.size < 2**64 or self._args.size % width:
                raise ValueError('Size must be a positive multiple of {} bytes'.format(width))
            configs = self._kernel_configs()
            size, dtype = self._args.size, self._args.data_type
            commands, targets = [], []
            if not configs:
                commands.append(self._command(dtype, 'all', size))
                targets.append({'STREAM_{}_{}'.format(op.upper(), dtype) for op in _OPERATIONS})
            else:
                for operation in _OPERATIONS:
                    tag = 'STREAM_{}_{}'.format(operation.upper(), dtype)
                    commands.append(self._command(dtype, operation, size, configs.get(tag)))
                    targets.append({tag})
            self._commands, self._command_targets = commands, targets
        except ValueError as error:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Invalid GPU STREAM configuration: {}.'.format(error))
            return False
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Read only canonical native winners; launch details stay in raw output."""
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)
        try:
            if not 0 <= cmd_idx < len(self._commands):
                raise ValueError('Unexpected command index')
            results = {}
            for line in raw_output.splitlines():
                if not line.startswith('STREAM_'):
                    continue
                fields = line.split()
                if len(fields) != 3 or _TAG.fullmatch(fields[0]) is None or fields[0] in results:
                    raise ValueError('Invalid or duplicate canonical result')
                tag, bandwidth, ratio = fields
                bandwidth, ratio = float(bandwidth), float(ratio)
                if (not math.isfinite(bandwidth) or bandwidth <= 0 or not math.isfinite(ratio) or
                        (ratio < 0 and ratio != -1)):
                    raise ValueError('Invalid bandwidth/efficiency for {}'.format(tag))
                results[tag] = bandwidth, ratio
            if set(results) != self._command_targets[cmd_idx]:
                raise ValueError('Missing or unexpected workload results')
            for tag, (bandwidth, ratio) in sorted(results.items()):
                self._result.add_result(tag + '_bw', bandwidth)
                self._result.add_result(tag + '_ratio', ratio)
        except ValueError as error:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error('Invalid GPU STREAM output: {}.'.format(error))
            return False
        return True


BenchmarkRegistry.register_benchmark('gpu-stream', GpuStreamBenchmark, platform=Platform.CUDA)
