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
        cls.createMockFiles(cls, ['bin/hpl_run.sh'])
        return True

    @decorator.load_data('tests/data/hpl_results.log')
    def test_hpl(self, results):
        """Test HPL benchmark command generation."""
        benchmark_name = 'cpu-hpl'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        parameters = '--cpu_arch zen3 \
        --blockSize 224 --coreCount 60 --blocks 1 --problemSize 224000'

        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess(hpl_template='third_party/hpl-tests/template_hpl.dat')
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.

        assert (benchmark._args.cpu_arch == 'zen3')
        assert (benchmark._args.blockSize == 224)
        assert (benchmark._args.coreCount == 60)
        assert (benchmark._args.blocks == 1)
        assert (benchmark._args.problemSize == 224000)

        # Check command
        assert (1 == len(benchmark._commands))
        assert ('60' in benchmark._commands[0])
        assert ('hpl_run.sh' in benchmark._commands[0])
        assert ('xhpl_z3' in benchmark._commands[0])

        # Check results
        assert (benchmark._process_raw_result(0, results))
        assert (benchmark.result['return_code'][0] == 0)
        assert (float(benchmark.result['time'][0]) == 4645.37)
        assert (float(benchmark.result['throughput'][0]) == 8126.1)


if __name__ == '__main__':
    unittest.main()
