# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""GPU STREAM bandwidth with chip presets and a fixed fallback sweep."""

import json
import math
import os
from pathlib import Path
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
        self._parser.add_argument('--gpu_name', type=str.lower, default=None,
                                 help='GPU preset key (h100, gb300, vr). Omit to autotune.')
        self._parser.add_argument('--size', type=int, default=None,
                                 help='Bytes per array; overrides the chip default without changing launch presets.')
        self._parser.add_argument('--num_warm_up', type=int, default=20)
        self._parser.add_argument('--num_loops', type=int, default=100)
        self._parser.add_argument('--data_type', choices=['float', 'double', 'both'], default='both')
        self._parser.add_argument('--check_data', action='store_true',
                                 help='Validate output. Requires 6x buffer size in host memory; use small sizes.')

    @staticmethod
    def _check_config(tag, config):
        """Check only the parameters needed to launch a preset."""
        match = _TAG.fullmatch(tag)
        if match is None or not isinstance(config, dict):
            raise ValueError('Invalid chip configuration key: {}'.format(tag))
        if set(config) != {'block_size', 'unroll'} or any(type(value) is not int for value in config.values()):
            raise ValueError('{} needs integer block_size and unroll'.format(tag))
        if config['block_size'] not in _BLOCKS or config['unroll'] not in _UNROLLS:
            raise ValueError('Unsupported block size/unroll for {}'.format(tag))

    def _gpu_config(self):
        """Select only the GPU name supplied by the user."""
        gpu_name = self._args.gpu_name
        if gpu_name is None:
            return {}
        if not gpu_name.strip():
            raise ValueError('gpu_name must not be empty')
        with Path(__file__).with_name('gpu_stream_profiles.json').open(encoding='utf-8') as stream:
            profiles = json.load(stream)
        if not isinstance(profiles, dict) or any(not isinstance(value, dict) for value in profiles.values()):
            raise ValueError('Expected a chip-name to configuration mapping')
        if gpu_name not in profiles:
            logger.warning('No GPU STREAM preset for {}; using the fixed sweep.'.format(gpu_name))
            return {}
        profile = profiles[gpu_name]
        if set(profile) != {'buffer_size', 'kernels'} or not isinstance(profile['kernels'], dict):
            raise ValueError('Each chip needs one buffer_size and a kernels mapping')
        size = profile['buffer_size']
        if type(size) is not int or not 0 < size < 2**64 or size % 8:
            raise ValueError('Chip buffer_size must be a positive multiple of 8 bytes')
        configs = profile['kernels']
        for tag, config in configs.items():
            self._check_config(tag, config)
        logger.info('GPU STREAM {}: {}.'.format(gpu_name, 'preset' if configs else 'fallback sweep'))
        return profile

    def _command(self, dtype, operation, size, config=None):
        """Use singleton native parameters for presets, or the native fixed sweep."""
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
        """Resolve size once, independently of chip launch presets."""
        if not super()._preprocess():
            return False
        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)
        self._commands, self._command_targets = [], []
        try:
            if (self._args.num_warm_up < 0 or self._args.num_loops <= 0 or
                    self._args.num_warm_up + self._args.num_loops >= 2**64):
                raise ValueError('Invalid warm-up/loop counts')
            width = 4 if self._args.data_type == 'float' else 8
            if self._args.size is not None and (not 0 < self._args.size < 2**64 or self._args.size % width):
                raise ValueError('Size must be a positive multiple of {} bytes'.format(width))
            profile = self._gpu_config()
            configs = profile.get('kernels', {})
            size = self._args.size if self._args.size is not None else profile.get('buffer_size', _DEFAULT_SIZE)
            dtypes = ['float', 'double'] if self._args.data_type == 'both' else [self._args.data_type]
            commands, targets = [], []
            if not configs:
                commands.append(self._command(self._args.data_type, 'all', size))
                targets.append({'STREAM_{}_{}'.format(op.upper(), dtype) for dtype in dtypes for op in _OPERATIONS})
            else:
                for dtype in dtypes:
                    for operation in _OPERATIONS:
                        tag = 'STREAM_{}_{}'.format(operation.upper(), dtype)
                        config = configs.get(tag)
                        commands.append(self._command(dtype, operation, size, config))
                        targets.append({tag})
            self._commands, self._command_targets = commands, targets
        except (OSError, ValueError) as error:
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
