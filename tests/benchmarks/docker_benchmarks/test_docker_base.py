# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DockerBenchmark modules."""

import re

from tests.helper import decorator
from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.docker_benchmarks import DockerBenchmark, CudaDockerBenchmark, RocmDockerBenchmark


class FakeDockerBenchmark(DockerBenchmark):
    """Fake benchmark inherit from DockerBenchmark."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the docker-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)
        pattern = r'\d+\.\d+'
        result = re.findall(pattern, raw_output)
        if len(result) != 2:
            return False

        try:
            result = [float(item) for item in result]
        except BaseException:
            return False

        self._result.add_result('cost1', result[0])
        self._result.add_result('cost2', result[1])

        return True


class FakeCudaDockerBenchmark(CudaDockerBenchmark):
    """Fake benchmark inherit from CudaDockerBenchmark."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the docker-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        return True


class FakeRocmDockerBenchmark(RocmDockerBenchmark):
    """Fake benchmark inherit from RocmDockerBenchmark."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the docker-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        return True


@decorator.cuda_test
def test_docker_benchmark_base():
    """Test DockerBenchmark."""
    # Negative case - DOCKERBENCHMARK_IMAGE_NOT_SET.
    benchmark = FakeDockerBenchmark('fake')
    assert (benchmark._benchmark_type == BenchmarkType.DOCKER)
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.DOCKERBENCHMARK_IMAGE_NOT_SET)

    # Negative case - DOCKERBENCHMARK_CONTAINER_NOT_SET.
    benchmark = FakeDockerBenchmark('fake')
    benchmark._image_uri = 'image'
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.DOCKERBENCHMARK_CONTAINER_NOT_SET)

    # Negative case - DOCKERBENCHMARK_IMAGE_PULL_FAILURE.
    benchmark = FakeDockerBenchmark('fake')
    benchmark._image_uri = 'image'
    benchmark._container_name = 'container'
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.DOCKERBENCHMARK_IMAGE_PULL_FAILURE)

    # Positive case
    benchmark = FakeDockerBenchmark('fake')
    benchmark._image_uri = 'ubuntu'
    benchmark._container_name = 'fake-docker-benchmark-test'
    benchmark._entrypoint = 'echo'
    benchmark._cmd = '-n "cost1: 10.2, cost2: 20.2"'
    benchmark.run()
    assert (
        benchmark._commands[0] ==
        'docker run -i --rm --privileged --net=host --ipc=host --name=fake-docker-benchmark-test'
        '  --entrypoint echo ubuntu -n "cost1: 10.2, cost2: 20.2"'
    )

    # Test for _platform_options.
    benchmark = FakeCudaDockerBenchmark('fake')
    assert (benchmark._platform_options == '--gpus=all')
    benchmark = FakeRocmDockerBenchmark('fake')
    assert (benchmark._platform_options == '--security-opt seccomp=unconfined --group-add video')
