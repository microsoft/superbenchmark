# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ib-traffic benchmark."""

import os
import numbers
import unittest
from pathlib import Path
from unittest import mock
from collections import defaultdict

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType, ReturnCode


class IBBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for IBBenchmark benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/ib_validation'])

    @classmethod
    def tearDownClass(cls):
        """Hook method for deconstructing the class fixture after running all tests in the class."""
        p = Path('hostfile')
        if p.is_file():
            p.unlink()
        super().tearDownClass()

    def test_generate_config(self):    # noqa: C901
        """Test util functions ."""
        test_config_file = 'test_gen_config.txt'

        def read_config(filename):
            config = []
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    pairs = line.strip().split(';')
                    config.append(pairs)
            return config

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
        # Small scale test
        node_num = 4
        for m in ['one-to-one', 'one-to-many', 'many-to-one']:
            benchmark.gen_traffic_pattern(node_num, m, test_config_file)
            config = read_config(test_config_file)
            assert (config == expected_config[m])
        # Large scale test
        node_num = 1000
        # check for 'one-to-many' and 'many-to-one'
        # In Nth step, the count of N is (N-1), others are all 1
        for m in ['one-to-many', 'many-to-one']:
            benchmark.gen_traffic_pattern(node_num, m, test_config_file)
            config = read_config(test_config_file)
            assert (len(config) == node_num)
            assert (len(config[0]) == node_num - 1)
            for step in range(node_num):
                server = defaultdict(int)
                client = defaultdict(int)
                for pair in config[step]:
                    pair = pair.split(',')
                    server[int(pair[0])] += 1
                    client[int(pair[1])] += 1
                for i in range(node_num):
                    if m == 'many-to-one':
                        if i == step:
                            assert (server[i] == node_num - 1)
                        else:
                            assert (client[i] == 1)
                    elif m == 'one-to-many':
                        if i == step:
                            assert (client[i] == node_num - 1)
                        else:
                            assert (server[i] == 1)
        # check for 'one-to-one'
        # Each index appears 1 time in each step
        # Each index has been combined once with all the remaining indexes
        benchmark.gen_traffic_pattern(node_num, 'one-to-one', test_config_file)
        config = read_config(test_config_file)
        if node_num % 2 == 1:
            assert (len(config) == node_num)
            assert (len(config[0]) == node_num // 2)
        else:
            assert (len(config) == node_num - 1)
            assert (len(config[0]) == node_num // 2)
        test_pairs = defaultdict(list)
        for step in range(len(config)):
            node = defaultdict(int)
            for pair in config[step]:
                pair = pair.split(',')
                node[int(pair[0])] += 1
                node[int(pair[1])] += 1
                test_pairs[int(pair[0])].append(int(pair[1]))
                test_pairs[int(pair[1])].append(int(pair[0]))
            for index in node:
                assert (node[index] == 1)
        for node in range(node_num):
            assert (sorted(test_pairs[node]) == [(i) for i in range(node_num) if i != node])

        Path(test_config_file).unlink()

    @mock.patch('superbench.common.utils.network.get_ib_devices')
    def test_ib_traffic_performance(self, mock_ib_devices):
        """Test ib-traffic benchmark."""
        # Test without ib devices
        # Check registry.
        benchmark_name = 'ib-traffic'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Check preprocess
        # Negative cases
        parameters = '--ib_index 0 --iters 2000 --pattern one-to-one --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_MPI_INIT_FAILURE)

        hosts = ['node0\n', 'node1\n', 'node2\n', 'node3\n']
        with open('hostfile', 'w') as f:
            f.writelines(hosts)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '4'
        parameters = '--ib_index 0 --iters 2000 --pattern one-to-one --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)

        # Positive cases
        os.environ['OMPI_COMM_WORLD_SIZE'] = '3'
        parameters = '--ib_index 0 --iters 2000 --pattern one-to-one --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = ['mlx5_0']
        ret = benchmark._preprocess()
        assert (ret is True)

        # Generate config
        parameters = '--ib_index 0 --iters 2000 --msg_size 33554432 --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '4'
        mock_ib_devices.return_value = ['mlx5_0']
        ret = benchmark._preprocess()
        Path('config.txt').unlink()
        assert (ret)
        expect_command = 'ib_validation --hostfile hostfile --cmd_prefix "ib_write_bw -F ' + \
            '--iters=2000 -d mlx5_0 -s 33554432" --input_config ' + os.getcwd() + '/config.txt'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        # Custom config
        config = ['0,1', '1,0;0,1', '0,1;1,0', '1,0;0,1']
        with open('test_config.txt', 'w') as f:
            for line in config:
                f.write(line + '\n')
        parameters = '--ib_index 0 --iters 2000 --msg_size 33554432 --config test_config.txt --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '2'
        mock_ib_devices.return_value = ['mlx5_0']
        ret = benchmark._preprocess()
        Path('test_config.txt').unlink()
        assert (ret)
        expect_command = 'ib_validation --hostfile hostfile --cmd_prefix "ib_write_bw -F ' + \
            '--iters=2000 -d mlx5_0 -s 33554432" --input_config test_config.txt'

        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)
        raw_output_0 = """
The predix of cmd to run is: ib_write_bw -a -d ibP257p0s0
Load the config file from: config.txt
Output will be saved to:
config:
0,1
1,0;0,1
0,1;1,0
1,0;0,1
config end
results from rank ROOT_RANK:
23452.6,
22212.6,22433
22798.8,23436.3
23435.3,22766.5
"""
        raw_output_1 = """
The predix of cmd to run is: ib_write_bw -F --iters=2000 -d mlx5_0 -s 33554432
Load the config file from: config.txt
Output will be saved to:
config:
0,1
1,0;0,1
0,1;1,0
1,0;0,1
config end
results from rank ROOT_RANK:
23452.6,
22212.6,22433,
22798.8,23436.3,
"""
        raw_output_2 = """
--------------------------------------------------------------------------
mpirun was unable to launch the specified application as it could not access
or execute an executable:

while attempting to start process rank 0.
--------------------------------------------------------------------------
2 total processes failed to start
"""

        # Check function process_raw_data.
        # Positive case - valid raw output.
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        assert (benchmark._process_raw_result(0, raw_output_0))

        for metric in benchmark.result:
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))
        # Negative case - valid raw output.
        assert (benchmark._process_raw_result(0, raw_output_1) is False)
        assert (benchmark._process_raw_result(0, raw_output_2) is False)
        os.environ.pop('OMPI_COMM_WORLD_RANK')

        # Check basic information.
        assert (benchmark.name == 'ib-traffic')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'ib_validation')

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.ib_index == 0)
        assert (benchmark._args.iters == 2000)
        assert (benchmark._args.msg_size == 33554432)
        assert (benchmark._args.commands == ['ib_write_bw'])
