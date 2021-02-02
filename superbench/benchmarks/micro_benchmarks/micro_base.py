# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the micro-benchmark base class."""

from abc import abstractmethod

from superbench.benchmarks import BenchmarkType, BenchmarkResult
from superbench.benchmarks.base import Benchmark


class MicroBenchmark(Benchmark):
    """The base class of micro-benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)
        self.__commands = list()

    def add_parser_auguments(self):
        """Add the specified auguments."""
        super().add_parser_auguments()

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking."""
        super()._preprocess()
        self._result = BenchmarkResult(self._name, BenchmarkType.MICRO.value, run_count=self._args.run_count)

    @abstractmethod
    def _benchmarking(self):
        """Implementation for benchmarking."""
        pass

    def _process_result(self, output):
        """Function to process raw results and save the summarized results."""
        pass

    def print_env_info(self):
        """Print environments or dependencies information."""
        pass
