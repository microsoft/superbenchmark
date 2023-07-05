# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DirectXGPUCopyBw performance benchmarks."""

import os
from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import MemBwBenchmark


class DirectXGPUCopyBw(MemBwBenchmark):
    """The GPUCopyBw benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._mem_types = ['htod', 'dtoh']
        self._bin_name = 'DirectXGPUCopyBw.exe'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--size',
            type=int,
            required=False,
            default=None,
            help='Size of data for GPU copy.',
        )
        self._parser.add_argument(
            '--warm_up',
            type=int,
            required=False,
            default=20,
            help='Number of warm up copy times to run.',
        )
        self._parser.add_argument(
            '--num_loops',
            type=int,
            required=False,
            default=1000,
            help='Number of copy times to run.',
        )
        self._parser.add_argument(
            '--minbytes',
            type=int,
            required=False,
            default=64,
            help='Run size from min_size to max_size for GPU copy.',
        )
        self._parser.add_argument(
            '--maxbytes',
            type=int,
            required=False,
            default=8 * 1024 * 1024,
            help='Run size from min_size to max_size for GPU copy.',
        )
        self._parser.add_argument(
            '--check',
            action='store_true',
            help='Whether check data after copy.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        for mem_type in self._args.mem_type:
            # Prepare the command line.
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += f' --{mem_type}'
            command += ' --warm_up ' + str(self._args.warm_up)
            command += ' --num_loops ' + str(self._args.num_loops)
            if self._args.size is not None:
                command += ' --size ' + str(self._args.size)
            else:
                command += ' --minbytes ' + str(self._args.minbytes)
                command += ' --maxbytes ' + str(self._args.maxbytes)
            if self._args.check:
                command += ' --check'
            self._commands.append(command)
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output', raw_output, self._args.log_raw_data)

        try:
            lines = raw_output.splitlines()
            for line in lines:
                if 'GB' in line:
                    type = line.split()[0].strip(':')
                    size = int(line.strip().split()[1].strip('B'))
                    bw = float(line.strip().split()[2])
                    self._result.add_result(f'{type}_{size}_bw', bw)
                if 'error' in line.lower():
                    logger.error(
                        'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                            self._curr_run_index, self._name, raw_output
                        )
                    )
                    return False
            return True
        except Exception as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, exception: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False


BenchmarkRegistry.register_benchmark('directx-gpu-copy-bw', DirectXGPUCopyBw, platform=Platform.DIRECTX)
