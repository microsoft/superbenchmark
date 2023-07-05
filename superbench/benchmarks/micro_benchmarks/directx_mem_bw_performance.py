# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DirectXGPUMemBw performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class DirectXGPUMemBw(MicroBenchmarkWithInvoke):
    """The DirectXGPUMemBw benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._bin_name = 'DirectXGPUMemRwBw.exe'
        self._modes = ['read', 'write', 'readwrite']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--num_warm_up',
            type=int,
            default=0,
            required=False,
            help='Number of warm up rounds.',
        )
        self._parser.add_argument(
            '--num_loop',
            type=int,
            default=100,
            required=False,
            help='Number of loop times to measure the performance.',
        )
        self._parser.add_argument(
            '--size',
            type=int,
            default=None,
            required=False,
            help='Size of data for GPU copy.',
        )
        self._parser.add_argument(
            '--minbytes',
            type=int,
            default=4096,
            required=False,
            help='Lower data size bound to test.',
        )
        self._parser.add_argument(
            '--maxbytes',
            type=int,
            default=1024 * 1024 * 1024,
            required=False,
            help='Upper data size bound to test.',
        )
        self._parser.add_argument(
            '--check_data',
            action='store_true',
            required=False,
            help='Whether check data correctness.',
        )
        self._parser.add_argument(
            '--mode',
            type=str,
            nargs='+',
            default=list(),
            help='Memory operation mode. E.g. {}.'.format(' '.join(self._modes)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking."""
        if not super()._preprocess():
            return False

        self._args.mode = [m.lower() for m in self._args.mode]
        for mode in self._args.mode:
            if mode not in self._modes:
                logger.warning(
                    'Unsupported mode - benchmark: {}, mode: {}, expected: {}.'.format(self._name, mode, self._modes)
                )
                self._args.mode.remove(mode)

        if len(self._args.mode) == 0:
            logger.error('No valid operation modes are provided.')
            return False

        for mode in self._args.mode:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += (' --num_warm_up ' + str(self._args.num_warm_up))
            command += (' --num_loop ' + str(self._args.num_loop))
            if self._args.size is not None:
                command += (' --size ' + str(self._args.size))
            else:
                command += (' --minbytes ' + str(self._args.minbytes))
                command += (' --maxbytes ' + str(self._args.maxbytes))
            if self._args.check_data:
                command += (' --check_data')
            command += (' --' + mode)
            self._commands.append(command)
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

        self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        mode = self._args.mode[cmd_idx]
        self._result.add_raw_data('raw_output_' + mode, raw_output, self._args.log_raw_data)

        valid = True

        content = raw_output.splitlines()
        try:
            for line in content:
                if 'GPUMemBw:' in line:
                    size = int(line.split()[-3])
                    bw = float(line.split()[-2])
                    self._result.add_result(f'{mode}_{size}_bw', bw)
                if 'error' in line.lower():
                    valid = False
        except BaseException:
            valid = False
        finally:
            if not valid:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False
        return True


BenchmarkRegistry.register_benchmark('directx-gpu-mem-bw', DirectXGPUMemBw, platform=Platform.DIRECTX)
