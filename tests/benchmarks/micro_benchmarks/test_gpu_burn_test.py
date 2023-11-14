# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gpu-burn benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class GpuBurnBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for gpu-burn benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/gpu_burn'])

    @decorator.load_data('tests/data/gpu_burn.log')
    def test_gpu_burn(self, results):
        """Test gpu-burn benchmark command generation."""
        benchmark_name = 'gpu-burn'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        time = 10

        parameters = '--doubles --tensor_core --time ' + str(time)
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.time == time)
        assert (benchmark._args.doubles)
        assert (benchmark._args.tensor_core)

        # Check command
        assert (1 == len(benchmark._commands))
        assert ('-d' in benchmark._commands[0])
        assert ('-tc' in benchmark._commands[0])
        assert (str(time) in benchmark._commands[0])

        # Check results
        assert (benchmark._process_raw_result(0, results))
        assert (benchmark.result['return_code'][0] == 0)
        assert (benchmark.result['time'][0] == time)
        for device in range(8):
            assert (benchmark.result['gpu_' + str(device) + '_pass'][0] == 1)
        assert (benchmark.result['abort'][0] == 0)
