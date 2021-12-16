# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nccl-bw benchmark."""

import numbers
import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class CudaNcclBwBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for CudaNcclBwBenchmark benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(
            cls, [
                f'bin/{name}' for name in [
                    'all_reduce_perf',
                    'all_gather_perf',
                    'broadcast_perf',
                    'reduce_perf',
                    'reduce_scatter_perf',
                    'alltoall_perf',
                ]
            ]
        )

    @decorator.load_data('tests/data/nccl_allgather.log')
    @decorator.load_data('tests/data/nccl_allreduce.log')
    @decorator.load_data('tests/data/nccl_reduce.log')
    @decorator.load_data('tests/data/nccl_broadcast.log')
    @decorator.load_data('tests/data/nccl_reducescatter.log')
    @decorator.load_data('tests/data/nccl_alltoall.log')
    def test_nccl_bw_performance(self, allgather, allreduce, reduce, broadcast, reducescatter, alltoall):
        """Test nccl-bw benchmark."""
        benchmark_name = 'nccl-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--ngpus 8')

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark)
        assert (benchmark.name == 'nccl-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.operation == 'allreduce')
        assert (benchmark._args.ngpus == 8)
        assert (benchmark._args.minbytes == '8')
        assert (benchmark._args.maxbytes == '8G')
        assert (benchmark._args.stepfactor == 2)
        assert (benchmark._args.check == 0)
        assert (benchmark._args.iters == 20)
        assert (benchmark._args.warmup_iters == 5)

        # Check command list
        bin_names = [
            'all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf',
            'alltoall_perf'
        ]

        command = bin_names[0] + benchmark._commands[0].split(bin_names[0])[1]
        expected_command = '{} -b 8 -e 8G -f 2 -g 8 -c 0 -n 20 -w 5'.format(bin_names[0])
        assert (command == expected_command)

        # Check results and metrics.
        # Case with no raw_output
        assert (benchmark._process_raw_result(0, '') is False)

        # Case with valid raw_output
        raw_output = {
            'allgather': allgather,
            'allreduce': allreduce,
            'reduce': reduce,
            'broadcast': broadcast,
            'reducescatter': reducescatter,
            'alltoall': alltoall,
        }

        for op in raw_output.keys():
            benchmark._args.operation = op
            assert (benchmark._process_raw_result(0, raw_output[op]))

            for name in ['time', 'algbw', 'busbw']:
                for size in ['8589934592', '4294967296', '2147483648', '1073741824', '536870912', '32']:
                    metric = op + '_' + size + '_' + name
                    assert (metric in benchmark.result)
                    assert (len(benchmark.result[metric]) == 1)
                    assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark.result['allreduce_8589934592_time'][0] == 63896.0)
        assert (benchmark.result['allreduce_8589934592_algbw'][0] == 134.44)
        assert (benchmark.result['allreduce_8589934592_busbw'][0] == 235.26)
        assert (benchmark.result['alltoall_8589934592_time'][0] == 33508.0)
        assert (benchmark.result['alltoall_8589934592_algbw'][0] == 256.36)
        assert (benchmark.result['alltoall_8589934592_busbw'][0] == 224.31)
