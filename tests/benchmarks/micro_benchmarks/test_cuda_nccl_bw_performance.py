# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nccl-bw benchmark."""

import os
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
        assert (benchmark._args.graph_iters == 0)
        assert (benchmark._args.in_place is False)
        assert (benchmark._args.data_type == 'float')

        # Check command list
        bin_names = [
            'all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf',
            'alltoall_perf'
        ]

        command = bin_names[0] + benchmark._commands[0].split(bin_names[0])[1]
        expected_command = '{} -b 8 -e 8G -f 2 -g 8 -c 0 -n 20 -w 5 -G 0 -d float'.format(bin_names[0])
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

        if 'SB_MODE_SERIAL_INDEX' in os.environ:
            os.environ.pop('SB_MODE_SERIAL_INDEX')
        if 'SB_MODE_PARALLEL_INDEX' in os.environ:
            os.environ.pop('SB_MODE_PARALLEL_INDEX')

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

        # Check with exec index info
        os.environ['SB_MODE_SERIAL_INDEX'] = '0'
        os.environ['SB_MODE_PARALLEL_INDEX'] = '0'
        exec_index = '0_0'

        for op in raw_output.keys():
            benchmark._args.operation = op
            assert (benchmark._process_raw_result(0, raw_output[op]))

            for name in ['time', 'algbw', 'busbw']:
                for size in ['8589934592', '4294967296', '2147483648', '1073741824', '536870912', '32']:
                    metric = op + '_' + exec_index + ':' + size + '_' + name
                    assert (metric in benchmark.result)
                    assert (len(benchmark.result[metric]) == 1)
                    assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark.result['allreduce_0_0:8589934592_time'][0] == 63896.0)
        assert (benchmark.result['allreduce_0_0:8589934592_algbw'][0] == 134.44)
        assert (benchmark.result['allreduce_0_0:8589934592_busbw'][0] == 235.26)
        assert (benchmark.result['alltoall_0_0:8589934592_time'][0] == 33508.0)
        assert (benchmark.result['alltoall_0_0:8589934592_algbw'][0] == 256.36)
        assert (benchmark.result['alltoall_0_0:8589934592_busbw'][0] == 224.31)

    @decorator.load_data('tests/data/nccl_allreduce.log')
    @decorator.load_data('tests/data/nccl_alltoall.log')
    def test_nccl_bw_performance_in_place_parsing(self, allreduce, alltoall):
        """Test nccl-bw benchmark in-place parsing."""
        benchmark_name = 'nccl-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--ngpus 8 --in_place')

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark._args.in_place is True)

        # Case with valid raw_output
        raw_output = {
            'allreduce': allreduce,
            'alltoall': alltoall,
        }

        if 'SB_MODE_SERIAL_INDEX' in os.environ:
            os.environ.pop('SB_MODE_SERIAL_INDEX')
        if 'SB_MODE_PARALLEL_INDEX' in os.environ:
            os.environ.pop('SB_MODE_PARALLEL_INDEX')

        for op in raw_output.keys():
            benchmark._args.operation = op
            assert (benchmark._process_raw_result(0, raw_output[op]))

            for name in ['time', 'algbw', 'busbw']:
                for size in ['8589934592', '4294967296', '2147483648', '1073741824', '536870912', '32']:
                    metric = op + '_' + size + '_' + name
                    assert (metric in benchmark.result)
                    assert (len(benchmark.result[metric]) == 1)
                    assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark.result['allreduce_8589934592_time'][0] == 63959.0)
        assert (benchmark.result['allreduce_8589934592_algbw'][0] == 134.30)
        assert (benchmark.result['allreduce_8589934592_busbw'][0] == 235.03)
        assert (benchmark.result['alltoall_8589934592_time'][0] == 33234.0)
        assert (benchmark.result['alltoall_8589934592_algbw'][0] == 258.47)
        assert (benchmark.result['alltoall_8589934592_busbw'][0] == 226.16)
