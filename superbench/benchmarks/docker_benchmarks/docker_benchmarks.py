# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod

from superbench.benchmarks.base import Benchmark


class DockerBenchmark(Benchmark):
    '''The base class of benchmarks packaged in docker container.

    Args:
        name: benchmark name.
        parameters: benchmark parameters.
    '''
    def __init__(self, name, parameters=''):
        super().__init__(name, parameters='')
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
