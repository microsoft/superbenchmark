# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the base class."""

import shlex
import signal
import traceback
import argparse
import numbers
from datetime import datetime
from operator import attrgetter
from abc import ABC, abstractmethod

import numpy as np

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.result import BenchmarkResult


class SortedMetavarTypeHelpFormatter(argparse.MetavarTypeHelpFormatter):
    """Custom HelpFormatter class for argparse which sorts option strings."""
    def add_arguments(self, actions):
        """Sort option strings before original add_arguments.

        Args:
            actions (argparse.Action): Argument parser actions.
        """
        super(SortedMetavarTypeHelpFormatter, self).add_arguments(sorted(actions, key=attrgetter('option_strings')))


class Benchmark(ABC):
    """The base class of all benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        self._name = name
        self._argv = list(filter(None, shlex.split(parameters))) if parameters is not None else list()
        self._benchmark_type = None
        self._parser = argparse.ArgumentParser(
            add_help=False,
            usage=argparse.SUPPRESS,
            allow_abbrev=False,
            formatter_class=SortedMetavarTypeHelpFormatter,
        )
        self._args = None
        self._curr_run_index = 0
        self._result = None

    def add_parser_arguments(self):
        """Add the specified arguments."""
        self._parser.add_argument(
            '--run_count',
            type=int,
            default=1,
            required=False,
            help='The run count of benchmark.',
        )
        self._parser.add_argument(
            '--duration',
            type=int,
            default=0,
            required=False,
            help='The elapsed time of benchmark in seconds.',
        )
        self._parser.add_argument(
            '--log_raw_data',
            action='store_true',
            default=False,
            help='Log raw data into file instead of saving it into result object.',
        )
        self._parser.add_argument(
            '--log_flushing',
            action='store_true',
            default=False,
            help='Real-time log flushing.',
        )

    def get_configurable_settings(self):
        """Get all the configurable settings.

        Return:
            All configurable settings in raw string.
        """
        return self._parser.format_help().strip()

    def parse_args(self, ignore_invalid=False):
        """Parse the arguments.

        Return:
            ret (bool): whether parse succeed or not.
            args (argparse.Namespace): parsed arguments.
            unknown (list): unknown arguments.
        """
        try:
            args, unknown = self._parser.parse_known_args(self._argv)
        except BaseException as e:
            if ignore_invalid:
                logger.info('Missing or invliad parameters, will ignore the error and skip the args checking.')
                return True, None, []
            else:
                logger.error('Invalid argument - benchmark: {}, message: {}.'.format(self._name, str(e)))
                return False, None, []

        ret = True
        if len(unknown) > 0:
            logger.error(
                'Unknown arguments - benchmark: {}, unknown arguments: {}'.format(self._name, ' '.join(unknown))
            )
            ret = False

        return ret, args, unknown

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        self.add_parser_arguments()
        ret, self._args, unknown = self.parse_args()

        if not ret:
            self._result = BenchmarkResult(self._name, self._benchmark_type, ReturnCode.INVALID_ARGUMENT)
            return False

        self._result = BenchmarkResult(
            self._name, self._benchmark_type, ReturnCode.SUCCESS, run_count=self._args.run_count
        )

        if not isinstance(self._benchmark_type, BenchmarkType):
            logger.error(
                'Invalid benchmark type - benchmark: {}, type: {}'.format(self._name, type(self._benchmark_type))
            )
            self._result.set_return_code(ReturnCode.INVALID_BENCHMARK_TYPE)
            return False

        return True

    def _postprocess(self):
        """Postprocess/cleanup operations after the benchmarking.

        Return:
            True if _postprocess() succeed.
        """
        return True

    @abstractmethod
    def _benchmark(self):
        """Implementation for benchmarking."""
        pass

    def run(self):
        """Function to launch the benchmarking.

        Return:
            True if run benchmark successfully.
        """
        ret = True
        self._start_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        try:
            ret &= self._preprocess()
            if ret:
                signal.signal(signal.SIGTERM, self.__signal_handler)
                for self._curr_run_index in range(self._args.run_count):
                    ret &= self._benchmark()
                if ret:
                    ret &= self.__check_result_format()
        except TimeoutError as e:
            self._result.set_return_code(ReturnCode.KILLED_BY_TIMEOUT)
            logger.error('Run benchmark failed - benchmark: %s, message: %s', self._name, e)
        except BaseException as e:
            self._result.set_return_code(ReturnCode.RUNTIME_EXCEPTION_ERROR)
            logger.error('Run benchmark failed - benchmark: {}, message: {}'.format(self._name, str(e)))
        else:
            ret &= self._postprocess()
        finally:
            self._end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            self._result.set_timestamp(self._start_time, self._end_time)

        return ret

    def __signal_handler(self, signum, frame):
        """Signal handler for benchmark.

        Args:
            signum (int): Signal number.
            frame (FrameType): Timeout frame.
        """
        logger.debug('Killed by %s', signal.Signals(signum).name)
        logger.debug(''.join(traceback.format_stack(frame, 5)))
        if signum == signal.SIGTERM:
            raise TimeoutError('Killed by SIGTERM or timeout!')

    def __check_result_format(self):
        """Check the validation of result object.

        Return:
            True if the result is valid.
        """
        if (not self.__check_result_type()) or (not self.__check_summarized_result()) or (not self.__check_raw_data()):
            self._result.set_return_code(ReturnCode.INVALID_BENCHMARK_RESULT)
            return False

        return True

    def __check_result_type(self):
        """Check the type of result object.

        Return:
            True if the result is instance of BenchmarkResult.
        """
        if not isinstance(self._result, BenchmarkResult):
            logger.error(
                'Invalid benchmark result type - benchmark: {}, type: {}'.format(self._name, type(self._result))
            )
            return False

        return True

    def __is_list_type(self, data, t):
        if isinstance(data, list) and all(isinstance(item, t) for item in data):
            return True
        return False

    def __is_list_list_type(self, data, t):
        if (self.__is_list_type(data, list) and all(isinstance(value, t) for item in data for value in item)):
            return True
        return False

    def __check_summarized_result(self):
        """Check the validation of summary result.

        Return:
            True if the summary result is instance of List[Number].
        """
        for metric in self._result.result:
            if not self.__is_list_type(self._result.result[metric], numbers.Number):
                logger.error(
                    'Invalid summarized result - benchmark: {}, metric: {}, result: {}.'.format(
                        self._name, metric, self._result.result[metric]
                    )
                )
                return False

        return True

    def __check_raw_data(self):
        """Check the validation of raw data.

        Return:
            True if the raw data is:
              instance of List[List[Number]] for BenchmarkType.MODEL.
              instance of List[str] for BenchmarkType.DOCKER.
              instance of List[List[Number]] or List[str] for BenchmarkType.MICRO.
        """
        for metric in self._result.raw_data:
            is_valid = True
            if self._benchmark_type == BenchmarkType.MODEL:
                is_valid = self.__is_list_list_type(self._result.raw_data[metric], numbers.Number)
            elif self._benchmark_type == BenchmarkType.DOCKER:
                is_valid = self.__is_list_type(self._result.raw_data[metric], str)
            elif self._benchmark_type == BenchmarkType.MICRO:
                is_valid = self.__is_list_type(self._result.raw_data[metric], str) or self.__is_list_list_type(
                    self._result.raw_data[metric], numbers.Number
                )
            if not is_valid:
                logger.error(
                    'Invalid raw data type - benchmark: {}, metric: {}, raw data: {}.'.format(
                        self._name, metric, self._result.raw_data[metric]
                    )
                )
                return False

        return True

    def _process_percentile_result(self, metric, result, reduce_type=None):
        """Function to process the percentile results.

        Args:
            metric (str): metric name which is the key.
            result (List[numbers.Number]): numerical result.
            reduce_type (ReduceType): The type of reduce function.
        """
        if len(result) > 0:
            percentile_list = ['50', '90', '95', '99', '99.9']
            for percentile in percentile_list:
                self._result.add_result(
                    '{}_{}'.format(metric, percentile),
                    np.percentile(result, float(percentile), interpolation='nearest'), reduce_type
                )

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass

    @property
    def name(self):
        """Decoration function to access benchmark name."""
        return self._result.name

    @property
    def type(self):
        """Decoration function to access benchmark type."""
        return self._result.type

    @property
    def run_count(self):
        """Decoration function to access benchmark run_count."""
        return self._result.run_count

    @property
    def return_code(self):
        """Decoration function to access benchmark return_code."""
        return self._result.return_code

    @property
    def start_time(self):
        """Decoration function to access benchmark start_time."""
        return self._result.start_time

    @property
    def end_time(self):
        """Decoration function to access benchmark end_time."""
        return self._result.end_time

    @property
    def raw_data(self):
        """Decoration function to access benchmark raw_data."""
        return self._result.raw_data

    @property
    def result(self):
        """Decoration function to access benchmark result."""
        return self._result.result

    @property
    def serialized_result(self):
        """Decoration function to access benchmark result."""
        return self._result.to_string()

    @property
    def default_metric_count(self):
        """Decoration function to get the count of default metrics."""
        return self._result.default_metric_count
