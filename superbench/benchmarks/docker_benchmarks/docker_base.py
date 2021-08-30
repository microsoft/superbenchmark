# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the docker-benchmark base class."""

import subprocess

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkType, ReturnCode
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
        self._benchmark_type = BenchmarkType.DOCKER

        # Command lines to launch the docker image and run the benchmarks inside docker.
        self._commands = list()

        # Image url of the current docker-benchmark.
        self._image = None

    '''
    # If need to add new arguments, super().add_parser_arguments() must be called.
    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
    '''

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if self._image is None:
            self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_IMAGE_NOT_SET)
            logger.error('The image url is not set - benchmark: {}.'.format(self._name))
            return False

        self._commands.append('docker pull {}'.format(self._image))

        return True

    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        for cmd_idx in range(len(self._commands)):
            logger.info(
                'Execute command - round: {}, benchmark: {}, command: {}.'.format(
                    self._curr_run_index, self._name, self._commands[cmd_idx]
                )
            )
            output = subprocess.run(
                self._commands[cmd_idx],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                check=False,
                universal_newlines=True
            )
            if output.returncode != 0:
                self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_EXECUTION_FAILURE)
                logger.error(
                    'Microbenchmark execution failed - round: {}, benchmark: {}, error message: {}.'.format(
                        self._curr_run_index, self._name, output.stdout
                    )
                )
                return False
            else:
                if not self._process_raw_result(cmd_idx, output.stdout):
                    self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_RESULT_PARSING_FAILURE)
                    return False

        return True

    def _process_raw_result(self, raw_output):
        """Function to process raw results and save the summarized results.

        Args:
            raw_output (str): raw output string of the docker benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        # TODO: will implement it when add real benchmarks in the future.
        return True

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass
