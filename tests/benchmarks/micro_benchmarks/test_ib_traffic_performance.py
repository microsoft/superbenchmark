# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ib-loopback benchmark."""

import os
import numbers
import unittest
from pathlib import Path
from unittest import mock

from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType, ReturnCode


class IBTrafficBenchmarkTest(unittest.TestCase):
    """Tests for IBTrafficBenchmark benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench'
        binary_path = Path(os.getenv('SB_MICRO_PATH'), 'bin')
        binary_path.mkdir(parents=True, exist_ok=True)
        self.__binary_file = Path(binary_path, 'ib_mpi')
        self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        self.__binary_file.unlink()

    def test_generate_config(self):
        """Test util functions ."""
        expected_config = {}
        expected_config['one-to-one'] = [['0,3', '1,2'], ['0,1', '2,3'], ['0,2', '3,1']]
        expected_config['many-to-one'] = [
            ['0,1', '0,2', '0,3'], ['1,0', '1,2', '1,3'], ['2,0', '2,1', '2,3'], ['3,0', '3,1', '3,2']
        ]
        expected_config['one-to-many'] = [
            ['1,0', '2,0', '3,0'], ['0,1', '2,1', '3,1'], ['0,2', '1,2', '3,2'], ['0,3', '1,3', '2,3']
        ]
        benchmark_name = 'ib-traffic'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)
        benchmark = benchmark_class(benchmark_name)
        node_num = 4
        for m in ['one-to-one', 'one-to-many', 'many-to-one']:
            config = []
            benchmark.gen_traffic_pattern(node_num, m, 'test_gen_config.txt')
            with open('test_gen_config.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    pairs = line.strip().split(';')
                    config.append(pairs)
            assert (config == expected_config[m])
        Path('test_gen_config.txt').unlink()

    @mock.patch('superbench.common.utils.network.get_ib_devices')
    def test_ib_loopback_all_sizes(self, mock_ib_devices):
        """Test ib-loopback benchmark for all sizes."""

        # Test without ib devices
        # Check registry.
        benchmark_name = 'ib-traffic'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Check preprocess
        # Negative case
        parameters = '--ib_index 0 --iters 2000 --pattern one-to-one'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MPI_INIT_FAILURE)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '3'
        parameters = '--ib_index 0 --iters 2000 --pattern one-to-one'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = ['mlx5_0']
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '4'
        parameters = '--ib_index 0 --iters 2000 --pattern one-to-one'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)

        # Positive case
        parameters = '--ib_index 0 --iters 2000 --msg_size 33554432'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '4'
        mock_ib_devices.return_value = ['mlx5_0']
        ret = benchmark._preprocess()
        Path('config.txt').unlink()
        assert (ret)

        expect_command = 'ib_mpi --cmd_prefix "ib_write_bw -F --iters=2000 -d mlx5_0 -s 33554432 -x 0" --input_config config.txt --output_path result.csv'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        config = ['0,1', '1,0;0,1', '0,1;1,0', '1,0;0,1']
        with open('test_config.txt', 'w') as f:
            for line in config:
                f.write(line + '\n')
        parameters = '--ib_index 0 --iters 2000 --msg_size 33554432 --config test_config.txt'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '2'
        mock_ib_devices.return_value = ['mlx5_0']
        ret = benchmark._preprocess()
        Path('test_config.txt').unlink()
        assert (ret)

        expect_command = 'ib_mpi --cmd_prefix "ib_write_bw -F --iters=2000 -d mlx5_0 -s 33554432 -x 0" --input_config test_config.txt --output_path result.csv'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)
        raw_output = """
The predix of cmd to run is: ib_write_bw -a -d ibP257p0s0
Load the config file from: config.txt
Output will be saved to: result.csv
config:
0,1
1,0;0,1
0,1;1,0
1,0;0,1
results from rank 0: 23452.6
results from rank 0: 22212.6 22433
results from rank 0: 22798.8 23436.3
results from rank 0: 23435.3 22766.5
"""
        output_file_content = """
command prefix: ib_write_bw -F --iters=2000 -d mlx5_0 -s 33554432 -x 0
config:
"(0,1)","(1,0)","(2,3)","(3,2)"
"(0,3)","(3,0)","(1,2)","(2,1)"
"(0,2)","(2,0)","(3,1)","(1,3)"
results:
23452.6,
22212.6,22433,
22798.8,23436.3,
23435.3,22766.5,
"""
        with open('result.csv', 'w') as f:
            f.write(output_file_content)
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        assert (benchmark._process_raw_result(0, raw_output))
        os.environ.pop('OMPI_COMM_WORLD_RANK')
        Path('result.csv').unlink()
        # Check function process_raw_data.
        # Positive case - valid raw output.
        for metric in benchmark.result:
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        # Check basic information.
        assert (benchmark.name == 'ib-traffic')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'ib_mpi')

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.ib_index == 0)
        assert (benchmark._args.iters == 2000)
        assert (benchmark._args.msg_size == 33554432)
        assert (benchmark._args.commands == ['ib_write_bw'])
