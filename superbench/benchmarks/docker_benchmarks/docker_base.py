# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the docker-benchmark base class."""

from abc import abstractmethod

from superbench.common.utils import logger, run_command
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

        # Image uri of the current docker-benchmark.
        self._image_uri = None

        # Container name of the current docker-benchmark.
        self._container_name = None

        # Default options for docker run.
        self._default_options = '-i --rm --privileged --net=host --ipc=host'

        # Platform-specific options for docker run.
        self._platform_options = None

        # Entrypoint option of the current docker-benchmark.
        self._entrypoint = None

        # CMD option of the current docker-benchmark.
        self._cmd = None

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

        if self._image_uri is None:
            self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_IMAGE_NOT_SET)
            logger.error('The image uri is not set - benchmark: {}.'.format(self._name))
            return False

        if self._container_name is None:
            self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_CONTAINER_NOT_SET)
            logger.error('The container name is not set - benchmark: {}.'.format(self._name))
            return False

        output = run_command('docker pull --quiet {}'.format(self._image_uri))
        if output.returncode != 0:
            self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_IMAGE_PULL_FAILURE)
            logger.error(
                'DockerBenchmark pull image failed - benchmark: {}, error message: {}.'.format(
                    self._name, output.stdout
                )
            )
            return False

        command = 'docker run '
        command += self._default_options
        command += ' --name={container_name} {platform_options} {entrypoint} {image} {cmd}'
        self._commands.append(
            command.format(
                container_name=self._container_name,
                platform_options=self._platform_options or '',
                entrypoint='' if self._entrypoint is None else '--entrypoint {}'.format(self._entrypoint),
                image=self._image_uri,
                cmd=self._cmd or ''
            )
        )

        return True

    def _postprocess(self):
        """Postprocess/cleanup operations after the benchmarking.

        Return:
            True if _postprocess() succeed.
        """
        rm_containers = 'docker stop --time 20 {container} && docker rm {container}'.format(
            container=self._container_name
        )
        run_command(rm_containers)

        rm_image = 'docker rmi {}'.format(self._image_uri)
        run_command(rm_image)

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
            output = run_command(self._commands[cmd_idx], flush_output=self._args.log_flushing)
            if output.returncode != 0:
                self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_EXECUTION_FAILURE)
                logger.error(
                    'DockerBenchmark execution failed - round: {}, benchmark: {}, error message: {}.'.format(
                        self._curr_run_index, self._name, output.stdout
                    )
                )
                return False
            else:
                if not self._process_raw_result(cmd_idx, output.stdout):
                    self._result.set_return_code(ReturnCode.DOCKERBENCHMARK_RESULT_PARSING_FAILURE)
                    return False

        return True

    @abstractmethod
    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the docker-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        pass

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass


class CudaDockerBenchmark(DockerBenchmark):
    """The base class of benchmarks packaged in nvidia docker container."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._platform_options = '--gpus=all'


class RocmDockerBenchmark(DockerBenchmark):
    """The base class of benchmarks packaged in amd docker container."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._platform_options = '--security-opt seccomp=unconfined --group-add video'
