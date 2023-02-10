# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for STREAM benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class CpuStreamBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for STREAM benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/streamZen3.exe'])
        return True

    @decorator.load_data('tests/data/streamResult.log')
    def test_stream(self, results):
        """Test STREAM benchmark command generation."""
        benchmark_name = 'cpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        cores = '0 4 8 12 16 20 24 28 30 34 38 42 46 50 54 58 60 64 68 72 76 80 84 88 90 94 98 102 106 110 114 118'
        coresList = [
            0, 4, 8, 12, 16, 20, 24, 28, 30, 34, 38, 42, 46, 50, 54, 58, 60, 64, 68, 72, 76, 80, 84, 88, 90, 94, 98,
            102, 106, 110, 114, 118
        ]
        arch = 'zen3'
        parameters = '--cpu_arch ' + arch + ' --cores ' + cores
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.cores == coresList)
        assert (benchmark._args.cpu_arch == arch)

        # Check command
        assert (1 == len(benchmark._commands))
        assert ('OMP_PLACES' in benchmark._commands[0])

        # Check results
        assert (benchmark._process_raw_result(0, results))
        assert (benchmark.result['return_code'][0] == 0)
        functions = ['copy', 'scale', 'add', 'triad']
        values = [342008.3, 342409.6, 343827.7, 363208.7]
        for index in range(0, 4):
            result = float(benchmark.result[functions[index] + '_throughput'][0])
            print(result, values[index])
            assert (result == values[index])
        assert (int(benchmark.result['threads'][0]) == 32)


if __name__ == '__main__':
    unittest.main()
