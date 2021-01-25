# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from superbench.benchmarks.benchmark_base import Benchmark


class MicroBenchmark(Benchmark):
    '''The base class of micro-benchmarks.

    Args:
        name: benchmark name.
        argv: benchmark parameters.
    '''
    def __init__(self, name, argv=''):
        super().__init__(name, argv='')
        self._commands = list()

    def add_parser_auguments(self):
        super().add_parser_auguments()

    def preprocess(self):
        super().preprocess()

    @abstractmethod
    def benchmarking(self):
        pass

    @abstractmethod
    def process_result(self, output):
        pass

    @abstractmethod
    def print_env_info(self):
        pass
