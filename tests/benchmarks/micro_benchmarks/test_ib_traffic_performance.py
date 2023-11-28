# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ib-traffic benchmark."""

import os
import numbers
import unittest
import uuid
from pathlib import Path
from unittest import mock
from collections import defaultdict
from superbench.common.utils import gen_topo_aware_config

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType, ReturnCode


def gen_hostlist(hostlist, num):
    """Generate a fake list of specified number of hosts."""
    hostlist.clear()
    for i in range(0, num):
        hostlist.append(str(uuid.uuid4()))


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
        for p in [Path('hostfile'), Path('config.txt')]:
            if p.is_file():
                p.unlink()
        super().tearDownClass()

    @decorator.load_data('tests/data/ib_traffic_topo_aware_hostfile')    # noqa: C901
    @decorator.load_data('tests/data/ib_traffic_topo_aware_expected_config')
    def test_generate_config(self, tp_hosts, tp_expected_config):    # noqa: C901
        """Test util functions ."""
        test_config_file = 'test_gen_config.txt'
        hostlist = []

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
        gen_hostlist(hostlist, node_num)
        for m in ['one-to-one', 'one-to-many', 'many-to-one']:
            benchmark.gen_traffic_pattern(hostlist, m, test_config_file)
            config = read_config(test_config_file)
            assert (config == expected_config[m])
        # Large scale test
        node_num = 1000
        gen_hostlist(hostlist, node_num)
        # check for 'one-to-many' and 'many-to-one'
        # In Nth step, the count of N is (N-1), others are all 1
        for m in ['one-to-many', 'many-to-one']:
            benchmark.gen_traffic_pattern(hostlist, m, test_config_file)
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
        benchmark.gen_traffic_pattern(hostlist, 'one-to-one', test_config_file)
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

        # check for 'topo-aware'
        # compare generated config file with pre-saved expected config file
        tp_ibstat_path = 'tests/data/ib_traffic_topo_aware_ibstat.txt'
        tp_ibnetdiscover_path = 'tests/data/ib_traffic_topo_aware_ibnetdiscover.txt'
        hostlist = tp_hosts.split()
        expected_config = tp_expected_config.split()
        config = gen_topo_aware_config(hostlist, tp_ibstat_path, tp_ibnetdiscover_path, 2, 6)
        assert (config == expected_config)

        Path(test_config_file).unlink()

    @mock.patch('superbench.common.devices.GPU.vendor', new_callable=mock.PropertyMock)
    def test_ib_traffic_performance(self, mock_gpu):
        """Test ib-traffic benchmark."""
        # Test without ib devices
        # Check registry.
        benchmark_name = 'ib-traffic'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Check preprocess
        # Negative cases
        parameters = '--ib_dev 0 --iters 2000 --pattern one-to-one --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        ret = benchmark._preprocess()
        assert (ret is False)
        # no hostfile
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

        hosts = ['node0\n', 'node1\n', 'node2\n', 'node3\n']
        with open('hostfile', 'w') as f:
            f.writelines(hosts)

        parameters = '--ib_dev 0 --msg_size invalid --pattern one-to-one --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

        # Positive cases
        os.environ['OMPI_COMM_WORLD_SIZE'] = '3'
        parameters = '--ib_dev 0 --iters 2000 --pattern one-to-one --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        ret = benchmark._preprocess()
        assert (ret is True)

        # Generate config
        parameters = '--ib_dev "$(echo mlx5_0)" --iters 2000 --msg_size 33554432 --hostfile hostfile'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '4'
        ret = benchmark._preprocess()
        Path('config.txt').unlink()
        assert (ret)
        expect_command = "ib_validation --send_cmd_prefix '" + benchmark._args.bin_dir + \
            "/ib_write_bw -F -n 2000 -d $(echo mlx5_0) -s 33554432 --report_gbits'" + \
            f" --recv_cmd_prefix '{benchmark._args.bin_dir}/ib_write_bw -F -n 2000" + \
            " -d $(echo mlx5_0) -s 33554432 --report_gbits' " + \
            f'--timeout 120 --hostfile hostfile --input_config {os.getcwd()}/config.txt'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        parameters = '--ib_dev mlx5_0 --msg_size 0 --iters 2000 --pattern one-to-one ' \
            + '--hostfile hostfile --gpu_dev 0 --direction gpu-to-gpu'
        mock_gpu.return_value = 'nvidia'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        ret = benchmark._preprocess()
        expect_command = "ib_validation --send_cmd_prefix '" + benchmark._args.bin_dir + \
            "/ib_write_bw -F -n 2000 -d mlx5_0 -a --use_cuda=0 --report_gbits'" + \
            f" --recv_cmd_prefix '{benchmark._args.bin_dir}/ib_write_bw -F -n 2000" + \
            " -d mlx5_0 -a --use_cuda=0 --report_gbits' " + \
            f'--timeout 120 --hostfile hostfile --input_config {os.getcwd()}/config.txt'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)
        mock_gpu.return_value = 'amd'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        ret = benchmark._preprocess()
        expect_command = expect_command.replace('cuda', 'rocm')
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        parameters = '--command ib_read_lat --ib_dev mlx5_0 --iters 2000 --msg_size 33554432 ' + \
            '--pattern one-to-one --hostfile hostfile --gpu_dev 0 --direction gpu-to-gpu'
        mock_gpu.return_value = 'nvidia'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        ret = benchmark._preprocess()
        expect_command = "ib_validation --send_cmd_prefix '" + benchmark._args.bin_dir + \
            "/ib_read_lat -F -n 2000 -d mlx5_0 -s 33554432 --use_cuda=0 --report_gbits'" + \
            f" --recv_cmd_prefix '{benchmark._args.bin_dir}/ib_read_lat -F -n 2000" + \
            " -d mlx5_0 -s 33554432 --use_cuda=0 --report_gbits' " + \
            f'--timeout 120 --hostfile hostfile --input_config {os.getcwd()}/config.txt'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        # Custom config
        config = ['0,1', '1,0;0,1', '0,1;1,0', '1,0;0,1']
        with open('test_config.txt', 'w') as f:
            for line in config:
                f.write(line + '\n')
        parameters = '--ib_dev mlx5_0 --timeout 180 --iters 2000 --msg_size 33554432 ' + \
            '--config test_config.txt --hostfile hostfile --direction cpu-to-cpu'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        os.environ['OMPI_COMM_WORLD_SIZE'] = '2'
        ret = benchmark._preprocess()
        Path('test_config.txt').unlink()
        assert (ret)
        expect_command = "ib_validation --send_cmd_prefix '" + benchmark._args.bin_dir + \
            "/ib_write_bw -F -n 2000 -d mlx5_0 -s 33554432 --report_gbits'" + \
            f" --recv_cmd_prefix '{benchmark._args.bin_dir}/ib_write_bw -F -n 2000" + \
            " -d mlx5_0 -s 33554432 --report_gbits' " + \
            '--timeout 180 --hostfile hostfile --input_config test_config.txt'

        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)
        # suppose gpu driver mismatch issue or other traffic issue cause -1 result
        raw_output_0 = """
The prefix of cmd to run is: ib_write_bw -a -d ibP257p0s0
Load the config file from: config.txt
Output will be saved to:
config:
0,1
1,0;0,1
0,1;1,0
1,0;0,1
config end
results from rank ROOT_RANK:
-1,
-1,-1
-1,-1
-1,-1
"""
        raw_output_1 = """
The prefix of cmd to run is: ib_write_bw -a -d ibP257p0s0
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
        raw_output_2 = """
The prefix of cmd to run is: ib_write_bw -F -n 2000 -d mlx5_0 -s 33554432
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
        raw_output_3 = """
--------------------------------------------------------------------------
mpirun was unable to launch the specified application as it could not access
or execute an executable:

while attempting to start process rank 0.
--------------------------------------------------------------------------
2 total processes failed to start
"""

        # Check function process_raw_data.
        # Positive cases - valid raw output.
        os.environ['OMPI_COMM_WORLD_RANK'] = '0'
        assert (benchmark._process_raw_result(0, raw_output_0))
        for metric in benchmark.result:
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))
        values = list(benchmark.result.values())[1:]
        assert (all(value == [-1.0] for value in values))

        assert (benchmark._process_raw_result(0, raw_output_1))
        for index, metric in enumerate(benchmark.result):
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1 if index == 0 else len(benchmark.result[metric]) == 2)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        # Negative cases - invalid raw output.
        assert (benchmark._process_raw_result(0, raw_output_2) is False)
        assert (benchmark._process_raw_result(0, raw_output_3) is False)
        os.environ.pop('OMPI_COMM_WORLD_RANK')

        # Check basic information.
        assert (benchmark.name == 'ib-traffic')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'ib_validation')

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.ib_dev == 'mlx5_0')
        assert (benchmark._args.iters == 2000)
        assert (benchmark._args.msg_size == [33554432])
        assert (benchmark._args.command == ['ib_write_bw'])
