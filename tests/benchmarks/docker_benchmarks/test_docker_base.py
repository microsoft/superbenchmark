# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DockerBenchmark modules."""

import os
import re

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.docker_benchmarks import DockerBenchmark


class FakeDockerBenchmark(DockerBenchmark):
    """Fake benchmark inherit from DockerBenchmark."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        command = os.path.join(self._args.bin_dir, self._bin_name)
        command += " -n 'cost1: 10.2, cost2: 20.2'"
        self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the docker-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output)
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


def test_docker_benchmark_base():
    """Test MicroBenchmarkWithInvoke."""
    # Negative case - DOCKERBENCHMARK_IMAGE_NOT_SET.
    benchmark = FakeDockerBenchmark('fake')
    assert (benchmark._benchmark_type == BenchmarkType.DOCKER)
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.DOCKERBENCHMARK_IMAGE_NOT_SET)

    # Negative case - DOCKERBENCHMARK_CONTAINER_NOT_SET.
    benchmark = FakeDockerBenchmark('fake')
    benchmark._image = 'image'
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.DOCKERBENCHMARK_CONTAINER_NOT_SET)

    # Negative case - DOCKERBENCHMARK_IMAGE_PULL_FAILURE.
    benchmark = FakeDockerBenchmark('fake')
    benchmark._image = 'image'
    benchmark._container = 'container'
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.DOCKERBENCHMARK_IMAGE_PULL_FAILURE)

    # Test for DockerBenchmark._benchmark().
    benchmark._commands.append("echo -n 'cost1: 10.2, cost2: 20.2'")
    benchmark._benchmark()
    assert (benchmark.raw_data['raw_output_0'] == ['cost1: 10.2, cost2: 20.2'])
    assert (benchmark.result['cost1'] == [10.2])
    assert (benchmark.result['cost2'] == [20.2])
