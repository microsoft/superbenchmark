# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the docker-benchmark base class."""

from abc import abstractmethod

from superbench.benchmarks import BenchmarkType
from superbench.benchmarks.base import Benchmark


class DockerBenchmark(Benchmark):
    """The base class of benchmarks packaged in docker container."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        # Command lines to launch the docker image and run the benchmarks inside docker.
        self.__commands = list()

    '''
    # If need to add new arguments, super().add_parser_arguments() must be called.
    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
    '''

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking."""
        super()._preprocess()
        self._result.set_benchmark_type(BenchmarkType.DOCKER.value)

    @abstractmethod
    def _benchmark(self):
        """Implementation for benchmarking."""
        pass

    def _process_docker_result(self, output):
        """Function to process raw results and save the summarized results."""
        pass

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass
