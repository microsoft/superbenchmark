# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the micro-benchmark base class."""

import os
import subprocess

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.base import Benchmark


class MicroBenchmark(Benchmark):
    """The base class of micro-benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._benchmark_type = BenchmarkType.MICRO
        # Set the binary directory.
        self._binary_dir = os.getenv('SUPERBENCH_BINARY_DIR', '/usr/local/bin')
        # Command lines to launch the micro-benchmarks.
        self._commands = list()

    '''
    # If need to add new arguments, super().add_parser_arguments() must be called.
    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
    '''

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        return super()._preprocess()

    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        for command in self._commands:
            logger.info(
                'Execute command - round: {}, benchmark: {}, command: {}.'.format(
                    self._curr_run_index, self._name, command
                )
            )
            output = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                check=False,
                universal_newlines=True
            )
            if output.returncode != 0:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
                logger.error(
                    'Microbenchmark execution failed - round: {}, benchmark: {}, error message: {}.'.format(
                        self._curr_run_index, self._name, output.stdout
                    )
                )
                return False
            else:
                if not self._process_raw_result(command, output.stdout):
                    self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
                    return False

        return True

    def _process_numeric_result(self, metric, result):
        """Function to save the numerical results.

        Args:
            metric (str): metric name which is the key.
            result (List[numbers.Number]): numerical result.

        Return:
            True if result list is not empty.
        """
        if len(result) == 0:
            logger.error(
                'Numerical result of benchmark is empty - round: {}, name: {}.'.format(
                    self._curr_run_index, self._name
                )
            )
            return False

        self._result.add_raw_data(metric, result)
        self._result.add_result(metric, sum(result) / len(result))
        return True

    def _process_raw_result(self, command, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            command (str): command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        # TODO: will implement it when add real benchmarks in the future.
        return True

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass
