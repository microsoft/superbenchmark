# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the base class."""

import argparse
import numbers
from datetime import datetime
from abc import ABC, abstractmethod

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkType
from superbench.benchmarks.result import BenchmarkResult


class Benchmark(ABC):
    """The base class of all benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        self._name = name
        self._argv = list(filter(None, parameters.split(' ')))
        self._benchmark_type = None
        self._parser = argparse.ArgumentParser(
            add_help=False,
            usage=argparse.SUPPRESS,
            allow_abbrev=False,
            formatter_class=argparse.MetavarTypeHelpFormatter
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

    def get_configurable_settings(self):
        """Get all the configurable settings.

        Return:
            All configurable settings in raw string.
        """
        return self._parser.format_help().strip()

    def parse_args(self):
        """Parse the arguments.

        Return:
            The parsed arguments and unknown arguments.
        """
        self._args, unknown = self._parser.parse_known_args(self._argv)
        if len(unknown) > 0:
            logger.warning(
                'Benchmark has unknown arguments, name: {}, unknown arguments: {}'.format(
                    self._name, ' '.join(unknown)
                )
            )

        return self._args, unknown

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking."""
        self.add_parser_arguments()
        self.parse_args()
        logger.log_assert(
            isinstance(self._benchmark_type, BenchmarkType),
            'Invalid benchmark type - name: {}, type: {}'.format(self._name, type(self._benchmark_type))
        )
        self._result = BenchmarkResult(self._name, self._benchmark_type.value, run_count=self._args.run_count)

    @abstractmethod
    def _benchmark(self):
        """Implementation for benchmarking."""
        pass

    def run(self):
        """Function to launch the benchmarking.

        Return:
            The serialized string of BenchmarkResult object.
        """
        self._preprocess()

        self._start_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        for self._curr_run_index in range(self._args.run_count):
            self._benchmark()
        self._end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        self._result.set_timestamp(self._start_time, self._end_time)
        self.check_result_format()

        return self._result.to_string()

    def check_result_format(self):
        """Check the type of result object.

        Return:
            True if the result is valid.
        """
        # Check if the result is instance of BenchmarkResult.
        logger.log_assert(
            isinstance(self._result, BenchmarkResult),
            'Type of result is invalid - expect: {}, got: {}.'.format(type(BenchmarkResult), type(self._result))
        )
        self.__check_summarized_result()
        self.__check_raw_data()

    def __check_summarized_result(self):
        # Check if the summary result is instance of List[Number].
        for metric in self._result.result:
            valid = isinstance(self._result.result[metric], list)
            if valid:
                for value in self._result.result[metric]:
                    valid |= isinstance(value, numbers.Number)

            logger.log_assert(
                valid,
                'Type of summarized result is invalid - benchmark: {}, metric name: {}, expect: List[Number], got: {}.'.
                format(self._name, metric, type(self._result.result[metric]))
            )

    def __check_raw_data(self):
        # Check if the raw data is:
        #   instance of List[List[Number]] for BenchmarkType.MODEL, and BenchmarkType.DOCKER.
        #   instance of List[str] for BenchmarkType.MICRO.
        for metric in self._result.raw_data:
            valid = isinstance(self._result.raw_data[metric], list)
            if valid:
                for run in self._result.raw_data[metric]:
                    if self._benchmark_type in [BenchmarkType.MODEL, BenchmarkType.DOCKER]:
                        valid = isinstance(run, list)
                        if not valid:
                            break
                        for value in run:
                            valid = isinstance(run, numbers.Number)

                    elif self._benchmark_type in [BenchmarkType.MICRO]:
                        valid = isinstance(run, str)

            logger.log_assert(
                valid, 'Type of raw data is invalid - benchmark: {}, metric name: {}, expect: {}, got: {}.'.format(
                    self._name, metric,
                    'List[str]' if self._benchmark_type == BenchmarkType.MICRO else 'List[List[Number]]',
                    type(self._result.raw_data[metric])
                )
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
