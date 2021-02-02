# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the base class."""

import argparse
from datetime import datetime
from abc import ABC, abstractmethod

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkResult


class Benchmark(ABC):
    """The base class of all benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        self._name = name
        self._argv = list(filter(None, parameters.split(' ')))
        self._parser = argparse.ArgumentParser(add_help=False, usage=argparse.SUPPRESS, allow_abbrev=False)
        self._args = None
        self._start_time = None
        self._end_time = None
        self._curr_index = 0
        self._result = None

    def add_parser_auguments(self):
        """Add the specified auguments."""
        self._parser.add_argument(
            '--run_count',
            type=int,
            default=1,
            metavar='',
            required=False,
            help='The run count of benchmark.',
        )
        self._parser.add_argument(
            '--duration',
            type=int,
            default=0,
            metavar='',
            required=False,
            help='The elapsed time of benchmark.',
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
        self.add_parser_auguments()
        self.parse_args()

    @abstractmethod
    def _benchmarking(self):
        """Implementation for benchmarking."""
        pass

    def run(self):
        """Function to launch the benchmarking.

        Return:
            The serialized string of BenchmarkResult object.
        """
        self._preprocess()

        self._start_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        for self._curr_index in range(self._args.run_count):
            self._benchmarking()
        self._end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        self._result.set_timestamp(self._start_time, self._end_time)
        self.check_result_format()

        return self._result.to_string()

    def check_result_format(self):
        """Check the type of result object.

        Return:
            True if the result is instance of BenchmarkResult.
        """
        assert(isinstance(self._result, BenchmarkResult)), \
            'Result type invalid, expect: {}, got: {}.'.format(
                type(BenchmarkResult), type(self._result))

    def print_env_info(self):
        """Print environments or dependencies information."""
        pass
