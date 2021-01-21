# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from superbench.benchmarks.benchmark_base import Benchmark


class DockerBenchmark(Benchmark):
    def __init__(self, name, argv=''):
        super().__init__(name, argv='')
        self._commands = list()

    def add_parser_auguments(self):
        super().add_parser_auguments()

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def benchmarking(self):
        pass

    @abstractmethod
    def process_result(self, output):
        pass

    @abstractmethod
    def print_env_info(self):
        pass
