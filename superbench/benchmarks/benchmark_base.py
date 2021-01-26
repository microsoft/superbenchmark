# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from datetime import datetime
from abc import ABC, abstractmethod
from superbench.common.benchmark_result import Result


class Benchmark(ABC):
    '''The base class of all benchmarks.

    Args:
        name: benchmark name.
        argv: benchmark parameters.
    '''

    def __init__(self, name, argv=''):
        self._name = name
        self._argv = list(filter(None, argv.split(' ')))
        self._parser = argparse.ArgumentParser(
            add_help=False,
            usage=argparse.SUPPRESS,
            allow_abbrev=False)
        self._args = None
        self._start_time = None
        self._end_time = None
        self._curr_index = 0
        self._result = None

    def add_parser_auguments(self):
        self._parser.add_argument(
            '--run_count', type=int, default=1, metavar='',
            required=False, help='The run count of benchmark.'
        )
        self._parser.add_argument(
            '--duration', type=int, default=0, metavar='',
            required=False, help='The elapsed time of benchmark.'
        )

    def get_configurable_settings(self):
        return self._parser.format_help().strip()

    def parse_args(self):
        self._args, unknown = self._parser.parse_known_args(self._argv)
        return self._args, unknown

    def preprocess(self):
        self.add_parser_auguments()
        self.parse_args()
        self._result = Result(name=self._name, run_count=self._args.run_count)

    @abstractmethod
    def benchmarking(self):
        pass

    def run(self):
        self.preprocess()

        self._start_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        for self._curr_index in range(self._args.run_count):
            self.benchmarking()
        self._end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        self._result.set_timestamp(self._start_time, self._end_time)
        self.check_result_format()

        return self._result.to_string()

    def check_result_format(self):
        assert(isinstance(self._result, Result)), \
            'Result type invalid, expect: {}, got: {}.'.format(
                type(Result), type(self._result))

    @abstractmethod
    def print_env_info(self):
        pass
