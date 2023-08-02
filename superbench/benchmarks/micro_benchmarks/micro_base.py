# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the micro-benchmark base class."""

import os
import shutil
import statistics
from abc import abstractmethod

from superbench.common.utils import logger, run_command
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

    @abstractmethod
    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        pass

    def _process_numeric_result(self, metric, result, reduce_type=None, cal_percentile=False):
        """Function to save the numerical results.

        Args:
            metric (str): metric name which is the key.
            result (List[numbers.Number]): numerical result.
            reduce_type (ReduceType): The type of reduce function.
            cal_percentile (bool): Whether to calculate the percentile results.

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

        self._result.add_raw_data(metric, result, self._args.log_raw_data)
        self._result.add_result(metric, statistics.mean(result), reduce_type)
        if cal_percentile:
            self._process_percentile_result(metric, result, reduce_type)

        return True

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass


class MicroBenchmarkWithInvoke(MicroBenchmark):
    """The base class of micro-benchmarks that need to invoke subprocesses."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        # Command lines to launch the micro-benchmarks.
        self._commands = list()

        # Binary name of the current micro-benchmark.
        self._bin_name = None

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--bin_dir',
            type=str,
            default=None,
            required=False,
            help='Specify the directory of the benchmark binary.',
        )
        self._parser.add_argument(
            '--tolerant_fail',
            action='store_true',
            default=False,
            help='Tolerant failure for sub microbenchmark.',
        )

    def _set_binary_path(self):
        """Search the binary from self._args.bin_dir or from system environment path and set the binary directory.

        If self._args.bin_dir is specified, the binary is only searched inside it. Otherwise, the binary is searched
        from system environment path.

        Return:
            True if the binary exists.
        """
        if self._bin_name is None:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_BINARY_NAME_NOT_SET)
            logger.error('The binary name is not set - benchmark: {}.'.format(self._name))
            return False

        self._args.bin_dir = shutil.which(self._bin_name, mode=os.X_OK, path=self._args.bin_dir)

        if self._args.bin_dir is None:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_BINARY_NOT_EXIST)
            logger.error(
                'The binary does not exist - benchmark: {}, binary name: {}, binary directory: {}.'.format(
                    self._name, self._bin_name, self._args.bin_dir
                )
            )
            return False

        self._args.bin_dir = os.path.dirname(self._args.bin_dir)

        return True

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Set the environment path.
        if os.getenv('SB_MICRO_PATH'):
            os.environ['PATH'] = os.path.join(os.getenv('SB_MICRO_PATH'), 'bin') + os.pathsep + os.getenv('PATH', '')
            os.environ['LD_LIBRARY_PATH'] = os.path.join(os.getenv('SB_MICRO_PATH'),
                                                         'lib') + os.pathsep + os.getenv('LD_LIBRARY_PATH', '')

        if not self._set_binary_path():
            return False

        return True

    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        ret = True
        for cmd_idx in range(len(self._commands)):
            logger.info(
                'Execute command - round: {}, benchmark: {}, command: {}.'.format(
                    self._curr_run_index, self._name, self._commands[cmd_idx]
                )
            )

            output = run_command(self._commands[cmd_idx], flush_output=self._args.log_flushing, cwd=self._args.bin_dir)
            if output.returncode != 0:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
                logger.error(
                    'Microbenchmark execution failed - round: {}, benchmark: {}, error message: {}.'.format(
                        self._curr_run_index, self._name, output.stdout
                    )
                )
                ret = False
            else:
                if not self._process_raw_result(cmd_idx, output.stdout):
                    self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
                    ret = False
            if not self._args.tolerant_fail and ret is False:
                return False

        return ret

    @abstractmethod
    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        pass
