# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for MicroBenchmark and MicroBenchmarkWithInvoke modules."""

import os
import re
import shutil

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmark, MicroBenchmarkWithInvoke


class FakeMicroBenchmark(MicroBenchmark):
    """Fake benchmark inherit from MicroBenchmark."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)

    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        return True


class FakeMicroBenchmarkWithInvoke(MicroBenchmarkWithInvoke):
    """Fake benchmark inherit from MicroBenchmarkWithInvoke."""
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
            raw_output (str): raw output string of the micro-benchmark.

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


def test_micro_benchmark_base():
    """Test MicroBenchmark."""
    benchmark = FakeMicroBenchmark('fake')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    benchmark._process_numeric_result('metric1', [1, 2, 3, 4, 5, 6])
    assert (benchmark.result['metric1'] == [3.5])
    assert (benchmark.raw_data['metric1'] == [[1, 2, 3, 4, 5, 6]])

    benchmark._result._BenchmarkResult__result = dict()
    benchmark._result._BenchmarkResult__raw_data = dict()
    benchmark._process_numeric_result('metric1', [1, 3, 4, 2, 6, 5], cal_percentile=True)
    assert (benchmark.result['metric1'] == [3.5])
    assert (benchmark.result['metric1_50'] == [3])
    assert (benchmark.result['metric1_90'] == [5])
    assert (benchmark.result['metric1_95'] == [6])
    assert (benchmark.result['metric1_99'] == [6])
    assert (benchmark.result['metric1_99.9'] == [6])
    assert (benchmark.raw_data['metric1'] == [[1, 3, 4, 2, 6, 5]])


def test_micro_benchmark_with_invoke_base():
    """Test MicroBenchmarkWithInvoke."""
    # Negative case - MICROBENCHMARK_BINARY_NAME_NOT_SET.
    benchmark = FakeMicroBenchmarkWithInvoke('fake')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_BINARY_NAME_NOT_SET)

    # Negative case - MICROBENCHMARK_BINARY_NOT_EXIST.
    benchmark = FakeMicroBenchmarkWithInvoke('fake')
    benchmark._bin_name = 'not_existed_binary'
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_BINARY_NOT_EXIST)

    # Positive case.
    benchmark = FakeMicroBenchmarkWithInvoke('fake')
    benchmark._bin_name = 'echo'
    assert (benchmark.run())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (os.path.join(benchmark._args.bin_dir, benchmark._bin_name) == shutil.which(benchmark._bin_name))
    assert (benchmark._commands[0] == (shutil.which(benchmark._bin_name) + " -n 'cost1: 10.2, cost2: 20.2'"))
    assert (benchmark.raw_data['raw_output_0'] == ['cost1: 10.2, cost2: 20.2'])
    assert (benchmark.result['cost1'] == [10.2])
    assert (benchmark.result['cost2'] == [20.2])
