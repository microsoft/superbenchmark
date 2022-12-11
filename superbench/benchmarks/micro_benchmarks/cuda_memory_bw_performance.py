# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Cuda memory performance benchmarks."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import MemBwBenchmark


class CudaMemBwBenchmark(MemBwBenchmark):
    """The Cuda memory performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'bandwidthTest'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--shmoo_mode',
            action='store_true',
            default=False,
            help='Enable shmoo mode for bandwidthtest.',
        )
        self._parser.add_argument(
            '--sleep',
            type=int,
            default=0,
            required=False,
            help='The seconds will be sleeped between each bandwidthtest command.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Check the arguments and generate the commands
        for mem_type in self._args.mem_type:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' --' + mem_type
            if self._args.shmoo_mode:
                command += ' mode=shmoo'
            if self._args.memory == 'pinned':
                command += ' memory=pinned'
            command += ' --csv'
            if self._args.sleep != 0:
                command += f' && sleep {self._args.sleep}'
            self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + self._args.mem_type[cmd_idx], raw_output, self._args.log_raw_data)

        mem_bw = -1
        valid = True
        content = raw_output.splitlines()
        try:
            metric = self._metrics[self._mem_types.index(self._args.mem_type[cmd_idx])]
            parse_logline = self._parse_logline_map[self._args.mem_type[cmd_idx]]
            for line in content:
                if parse_logline in line:
                    line = line.split(',')[1]
                    value = re.search(r'(\d+.\d+)', line)
                    if value:
                        mem_bw = max(mem_bw, float(value.group(0)))

        except BaseException:
            valid = False
        finally:
            if valid is False or mem_bw == -1:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False

        self._result.add_result(metric, mem_bw)

        return True


BenchmarkRegistry.register_benchmark('mem-bw', CudaMemBwBenchmark, platform=Platform.CUDA)
