# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for HPL benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class CpuHplBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for HPL benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/xpl'])
        return True

    @decorator.load_data('tests/data/hpl_result.log')
    def test_stream(self, results):
        """Test STREAM benchmark command generation."""
        benchmark_name = 'cpu-hpl'
        (benchmark_class,
            predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        parameters = ''
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        # assert (benchmark._args.cores == coresList)
        # assert (benchmark._args.cpu_arch == arch)

        # Check command
        # assert (1 == len(benchmark._commands))
        # assert ('OMP_PLACES' in benchmark._commands[0])

        # Check results
        assert (benchmark._process_raw_result(0, results))
        assert (benchmark.result['return_code'][0] == 0)

        assert (int(benchmark.result['time'][0]) == 4645.37)
        assert (int(benchmark.result['Gflops'][0]) == 8.1261e+03)


if __name__ == '__main__':
    unittest.main()