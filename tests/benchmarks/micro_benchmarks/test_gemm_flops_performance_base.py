# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GemmFlopsBenchmark modules."""

import os

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks import GemmFlopsBenchmark


class FakeGemmFlopsBenchmark(GemmFlopsBenchmark):
    """Fake benchmark inherit from GemmFlopsBenchmark."""
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
        for precision in self._precision_need_to_run:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' "--precision ' + precision
            command += ' --m ' + str(self._args.m)
            command += ' --n ' + str(self._args.n)
            command += ' --k ' + str(self._args.k)
            command += ' --num_warmup ' + str(self._args.num_warmup) + '"'
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
            params = raw_output.strip('\n').split('--')
            for param in params[1:]:
                key_value = param.split()
                if key_value[0] == 'precision':
                    if key_value[1] != self._precision_need_to_run[cmd_idx]:
                        return False
            metric = self._precision_need_to_run[cmd_idx]
        except BaseException:
            return False

        self._result.add_result(metric, 0)

        return True


def test_gemm_flops_performance_base():
    """Test GemmFlopsBenchmark."""
    # Positive case - memory=pinned.
    benchmark = FakeGemmFlopsBenchmark('fake')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    # Check command list
    expected_command = [
        'echo "--precision fp64 --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision fp32 --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision fp16 --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision fp64_tc --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision tf32_tc --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision bf16_tc --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision fp16_tc --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision int8_tc --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision int4_tc --m 16384 --n 16384 --k 16384 --num_warmup 5"'
    ]
    for i in range(len(expected_command)):
        command = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
        assert (command == expected_command[i])
    for i, metric in enumerate(
        ['fp64', 'fp32', 'fp16', 'fp64_tc', 'tf32_tc', 'bf16_tc', 'fp16_tc', 'int8_tc', 'int4_tc']
    ):
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)

    # Positive case - memory=unpinned.
    benchmark = FakeGemmFlopsBenchmark('fake', parameters='--precision fp64 fp32 fp16')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run())
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    # Check command list
    expected_command = [
        'echo "--precision fp64 --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision fp32 --m 16384 --n 16384 --k 16384 --num_warmup 5"',
        'echo "--precision fp16 --m 16384 --n 16384 --k 16384 --num_warmup 5"'
    ]
    for i in range(len(expected_command)):
        command = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
        assert (command == expected_command[i])
    for i, metric in enumerate(['fp64', 'fp32', 'fp16']):
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)

    benchmark = FakeGemmFlopsBenchmark('fake', parameters='--precision fp64 bf64')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run() is True)

    # Negative case - INVALID_ARGUMENT.
    benchmark = FakeGemmFlopsBenchmark('fake', parameters='--precision bf64')
    assert (benchmark._benchmark_type == BenchmarkType.MICRO)
    assert (benchmark.run() is False)
    assert (benchmark.return_code == ReturnCode.NO_SUPPORTED_PRECISION)
