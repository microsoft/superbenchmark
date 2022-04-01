# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for MemBwBenchmark modules."""

import os

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks import MemBwBenchmark


class FakeMemBwBenchmark(MemBwBenchmark):
    """Fake benchmark inherit from MemBwBenchmark."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)
        self._bin_name = 'echo'

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Check the arguments and generate the commands
        for mem_type in self._args.mem_type:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' "--' + mem_type
            if self._args.memory == 'pinned':
                command += ' memory=pinned'
            command += '"'
            self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        try:
            params = raw_output.strip('\n').split(' memory=')
            if params[0][2:] not in self._mem_types:
                return False
            if self._args.memory == 'pinned':
                if params[1] not in self._memory:
                    return False
            metric = self._metrics[self._mem_types.index(self._args.mem_type[cmd_idx])]
        except BaseException:
            return False

        self._result.add_result(metric, 0)

        return True


def test_memory_bw_performance_base():
    """Test MemBwBenchmark."""
    # Positive case - memory=pinned.
    benchmark = FakeMemBwBenchmark('fake')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    # Check command list
    expected_command = ['echo "--htod memory=pinned"', 'echo "--dtoh memory=pinned"', 'echo "--dtod memory=pinned"']
    for i in range(len(expected_command)):
        command = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
        assert (command == expected_command[i])
    for i, metric in enumerate(['h2d_bw', 'd2h_bw', 'd2d_bw']):
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)

    # Positive case - memory=unpinned.
    benchmark = FakeMemBwBenchmark('fake', parameters='--memory unpinned')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    # Check command list
    expected_command = ['echo "--htod"', 'echo "--dtoh"', 'echo "--dtod"']
    for i in range(len(expected_command)):
        command = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
        assert (command == expected_command[i])
    for i, metric in enumerate(['h2d_bw', 'd2h_bw', 'd2d_bw']):
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)

    # Negative case - INVALID_ARGUMENT.
    benchmark = FakeMemBwBenchmark('fake', parameters='--memory fake')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

    benchmark = FakeMemBwBenchmark('fake', parameters='--mem_type fake')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)
