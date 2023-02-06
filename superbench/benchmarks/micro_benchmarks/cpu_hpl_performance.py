# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module for running the HPL benchmark."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CpuHplBenchmark(MicroBenchmarkWithInvoke):
    """The HPL benchmark class."""

    def __init__(self, name, parameters=''):
        """Constructor.
        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'xhpl'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        # self._parser.add_argument(
        #     '--cpu_arch',
        #     type=str,
        #     default='zen4',
        #     required=False,
        #     help='The targeted cpu architectures to run \
        #         STREAM. Possible values are {}.'.format(' '.join(self.__cpu_arch))
        # )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.
        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if not self._set_binary_path():
            logger.error(
                'Executable {} not found in {} or it is not executable'.format(self._bin_name, self._args.bin_dir)
            )
            return False

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
        content = raw_output.splitlines()
        results
        for idx, line in enumerate(content):
            if "T/V" in line and 'Gflops' in line:
                break

        results = content[idx+2].split()

        for line in content[idx+2:]:
            if "1 tests completed and passed residual checks" in line:
                self._result.add_result("tests_pass", 1)
            elif "0 tests completed and passed residual checks" in line::
                self._result.add_result("tests_pass", 0)

        self._result.add_result( 'time',float(results[5]))
        self._result.add_result( 'Gflops',float(results[6]))

        # raw output
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        return True


BenchmarkRegistry.register_benchmark('cpu-hpl', CpuStreamBenchmark)