# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for disk-performance benchmark."""

from pathlib import Path
import os
import unittest

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class GpuSmCopyBwBenchmarkTest(unittest.TestCase):
    """Test class for gpu-sm-copy-bw benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench/'
        binary_path = Path(os.getenv('SB_MICRO_PATH'), 'bin')
        binary_path.mkdir(parents=True, exist_ok=True)
        self.__binary_file = binary_path / 'gpu_sm_copy'
        self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        self.__binary_file.unlink()

    def test_gpu_sm_copy_bw_empty_param(self):
        """Test gpu-sm-copy-bw benchmark command generation with empty parameter."""
        benchmark_name = 'gpu-sm-copy-bw'

        for platform in [Platform.CUDA, Platform.ROCM]:
            (benchmark_class,
             predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
            assert (benchmark_class)

            benchmark = benchmark_class(benchmark_name, parameters='')

            # Check basic information
            assert (benchmark)
            ret = benchmark._preprocess()
            assert (ret is True)
            assert (benchmark.return_code == ReturnCode.SUCCESS)
            assert (benchmark.name == 'gpu-sm-copy-bw')
            assert (benchmark.type == BenchmarkType.MICRO)

            # Command list should be empty
            assert (0 == len(benchmark._commands))

    def test_gpu_sm_copy_bw_benchmark_disabled(self):
        """Test gpu-sm-copy-bw benchmark command generation with all benchmarks disabled."""
        benchmark_name = 'gpu-sm-copy-bw'

        for platform in [Platform.CUDA, Platform.ROCM]:
            (benchmark_class,
             predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
            assert (benchmark_class)

            numa_nodes = ['1', '2']
            numa_node_option = '--numa_nodes ' + ' '.join(numa_nodes)

            gpu_ids = ['3', '4']
            gpu_id_option = '--gpu_ids ' + ' '.join(gpu_ids)

            param_str = numa_node_option + ' ' + gpu_id_option
            benchmark = benchmark_class(benchmark_name, parameters=param_str)

            # Check basic information
            assert (benchmark)
            ret = benchmark._preprocess()
            assert (ret is True)
            assert (benchmark.return_code == ReturnCode.SUCCESS)
            assert (benchmark.name == 'gpu-sm-copy-bw')
            assert (benchmark.type == BenchmarkType.MICRO)

            # Command list should be empty
            assert (0 == len(benchmark._commands))

    def test_gpu_sm_copy_bw_benchmark_enabled(self):
        """Test disk-performance benchmark command generation with all benchmarks enabled."""
        benchmark_name = 'gpu-sm-copy-bw'

        for platform in [Platform.CUDA, Platform.ROCM]:
            (benchmark_class,
             predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
            assert (benchmark_class)

            numa_nodes = ['1', '2']
            numa_node_option = '--numa_nodes ' + ' '.join(numa_nodes)

            gpu_ids = ['3', '4']
            gpu_id_option = '--gpu_ids ' + ' '.join(gpu_ids)

            param_str = numa_node_option + ' ' + gpu_id_option
            param_str += ' --enable_dtoh'
            param_str += ' --enable_htod'
            test_data_size = '5'
            param_str += ' --size=%s' % test_data_size
            test_num_loops = '6'
            param_str += ' --num_loops=%s' % test_num_loops
            benchmark = benchmark_class(benchmark_name, parameters=param_str)

            # Check basic information
            assert (benchmark)
            ret = benchmark._preprocess()
            assert (ret is True)
            assert (benchmark.return_code == ReturnCode.SUCCESS)
            assert (benchmark.name == 'gpu-sm-copy-bw')
            assert (benchmark.type == BenchmarkType.MICRO)

            # Check command list
            # 2 NUMA nodes * 2 GPU IDs * 2 directions = 8 commands
            assert (8 == len(benchmark._commands))

            # Check parameter assignments
            command_idx = 0
            for numa_node in numa_nodes:
                for gpu_id in gpu_ids:
                    for copy_direction in ['dtoh', 'htod']:
                        assert (
                            benchmark._commands[command_idx] == 'numactl -N %s -m %s %s %d %s %s %s' % (
                                numa_node, numa_node, self.__binary_file, gpu_id, copy_direction, test_data_size,
                                test_data_size
                            )
                        )
                        command_idx += 1

    def test_gpu_sm_copy_bw_result_parsing(self):
        """Test gpu-sm-copy-bw benchmark result parsing."""
        benchmark_name = 'gpu-sm-copy-bw'

        for platform in [Platform.CUDA, Platform.ROCM]:
            (benchmark_class,
             predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
            assert (benchmark_class)
            benchmark = benchmark_class(benchmark_name, parameters='--numa_nodes 0 --gpu_ids 0 --enable_dtoh')
            assert (benchmark)
            ret = benchmark._preprocess()
            assert (ret is True)
            assert (benchmark.return_code == ReturnCode.SUCCESS)
            assert (benchmark.name == 'gpu-sm-copy-bw')
            assert (benchmark.type == BenchmarkType.MICRO)

            result_key = 'gpu_sm_copy_performance:numa0:gpu0:dtoh'
            result_bw = '4.68247'

            # Positive case - valid raw output.
            test_raw_output = 'Bandwidth (GB/s): %s' % result_bw
            assert (benchmark._process_raw_result(0, test_raw_output))
            assert (benchmark.return_code == ReturnCode.SUCCESS)

            assert (1 == len(benchmark.result.keys()))
            assert (float(4.68247) == benchmark.result[result_key][0])

            # Negative case - invalid raw output.
            assert (benchmark._process_raw_result(1, 'Invalid raw output') is False)
            assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
