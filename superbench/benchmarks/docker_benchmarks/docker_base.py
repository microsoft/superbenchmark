# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the docker-benchmark base class."""

from abc import abstractmethod

from superbench.benchmarks.base import Benchmark


class DockerBenchmark(Benchmark):
    """The base class of benchmarks packaged in docker container."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters='')
        self._commands = list()

    def add_parser_auguments(self):
        """Add the specified auguments."""
        super().add_parser_auguments()

    def preprocess(self):
        """Preprocess/preparation operations before the benchmarking."""
        super().preprocess()

    @abstractmethod
    def benchmarking(self):
        """Implementation for benchmarking."""
        pass

    @abstractmethod
    def process_result(self, output):
        """Function to process raw results and save the summarized results.

        Args:
            output (str): raw output of the benchmarking.
        """
        pass

    def print_env_info(self):
        """Print environments or dependencies information."""
        pass
