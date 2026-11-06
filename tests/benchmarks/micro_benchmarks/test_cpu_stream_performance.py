# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for STREAM benchmark."""

import tempfile
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
        cls.createMockFiles(cls, ['bin/stream'])
        cls.createMockFiles(cls, ['bin/streamZen3'])
        cls.createMockFiles(cls, ['bin/streamNeo2'])
        cls.createMockFiles(cls, ['bin/streamOlympus'])
        cls.createMockFiles(cls, ['bin/streamNative'])
        return True

    def test_stream_core_selection(self):
        """Test explicit core lists and inclusive start,end,step ranges."""
        benchmark_name = 'cpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        cases = [
            ('88,151,1', list(range(88, 152))),
            ('88,151,2', list(range(88, 152, 2))),
            ('88,150,2', list(range(88, 151, 2))),
            ('88,88,1', [88]),
            ('0,3,2', [0, 2]),
            ('88,90,10', [88]),
            ('88 90 92', [88, 90, 92]),
            ('92 88 90', [92, 88, 90]),
            ('88', [88]),
        ]
        for cores, expected in cases:
            with self.subTest(cores=cores):
                benchmark = benchmark_class(
                    benchmark_name, parameters='--cpu_arch native --numa_mem_nodes 1 --cores ' + cores
                )
                assert (benchmark._preprocess() is True)
                assert (benchmark.return_code == ReturnCode.SUCCESS)
                assert (benchmark._args.cores == expected)
                assert (len(benchmark._commands) == 1)
                command = benchmark._commands[0].split()
                assert ('OMP_NUM_THREADS={}'.format(len(expected)) in command)
                assert ('OMP_PLACES=' + ','.join(f'{{{core}}}' for core in expected) in command)
                assert ('numactl -m1' in benchmark._commands[0])
                assert ('streamNative' in benchmark._commands[0])

    def test_stream_default_cores(self):
        """Test that the default core configuration is unchanged."""
        benchmark_name = 'cpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name)
        assert (benchmark._preprocess() is True)
        assert (
            benchmark._args.cores == [
                0, 8, 16, 24, 32, 38, 44, 52, 60, 68, 76, 82, 88, 96, 104, 112, 120, 126, 132, 140, 148, 156, 164, 170
            ]
        )

    def test_stream_invalid_cores(self):
        """Test that malformed core selections fail argument validation."""
        benchmark_name = 'cpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        for cores in [
            '', '88,151', '88,151,1,2', '88,,1', 'a,151,1', '88,151,a', '88,151,0', '88,151,-1', '151,88,1', '-1,151,1',
            '88,151,1 152', '88 90,100,2', '88 90 foo'
        ]:
            with self.subTest(cores=cores):
                benchmark = benchmark_class(benchmark_name, parameters='--cores ' + cores)
                assert (benchmark._preprocess() is False)
                assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)
                assert (not benchmark._commands)

    @decorator.load_data('tests/data/streamResultZen.log')
    def test_stream_zen(self, results):
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
        assert (benchmark._commands[0].startswith('env '))
        assert ('&&' not in benchmark._commands[0])
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

    @decorator.load_data('tests/data/streamResultNeo.log')
    def test_stream_neo(self, results):
        """Test STREAM benchmark command generation for neo2."""
        benchmark_name = 'cpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        cores = '0 1 2 3'
        coresList = [0, 1, 2, 3]
        arch = 'neo2'
        parameters = '--cpu_arch ' + arch + ' --cores ' + cores + ' --numa_mem_nodes 0'
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
        assert (benchmark._args.numa_mem_nodes == [0])

        # Check command
        assert (1 == len(benchmark._commands))
        assert ('OMP_PLACES' in benchmark._commands[0])
        assert ('numactl -m0' in benchmark._commands[0])

        # Check results
        assert (benchmark._process_raw_result(0, results))
        functions = ['copy', 'scale', 'add', 'triad']
        values = [108180.5, 108572.8, 104571.7, 105256.0]
        for index in range(0, 4):
            result = float(benchmark.result[functions[index] + '_throughput'][0])
            assert (result == values[index])

    def test_stream_arm64_binary_selection(self):
        """Test STREAM benchmark binary selection for ARM64 architectures."""
        benchmark_name = 'cpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        for arch, binary_name in [('olympus', 'streamOlympus'), ('native', 'streamNative')]:
            with self.subTest(cpu_arch=arch):
                benchmark = benchmark_class(benchmark_name, parameters='--cpu_arch ' + arch + ' --cores 0')

                assert (benchmark._preprocess() is True)
                assert (benchmark._args.cpu_arch == arch)
                assert (benchmark._bin_name == binary_name)
                assert (binary_name in benchmark._commands[0])

    def test_stream_missing_arm64_binary(self):
        """Test STREAM benchmark fails when the selected ARM64 binary is not built, e.g. no GCC 15 for Olympus."""
        benchmark_name = 'cpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        with tempfile.TemporaryDirectory() as bin_dir:
            benchmark = benchmark_class(
                benchmark_name, parameters='--cpu_arch olympus --cores 0 --bin_dir {}'.format(bin_dir)
            )
            assert (benchmark._preprocess() is False)
            assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_BINARY_NOT_EXIST)


if __name__ == '__main__':
    unittest.main()
