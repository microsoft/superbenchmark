# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner test."""

import json
import unittest
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import yaml
from omegaconf import OmegaConf

from superbench.runner import SuperBenchRunner


class RunnerTestCase(unittest.TestCase):
    """A class for runner test cases."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        default_config_file = Path(__file__).parent / '../../superbench/config/default.yaml'
        with default_config_file.open() as fp:
            self.default_config = OmegaConf.create(yaml.load(fp, Loader=yaml.SafeLoader))
        self.sb_output_dir = tempfile.mkdtemp()

        self.runner = SuperBenchRunner(self.default_config, None, None, self.sb_output_dir)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        shutil.rmtree(self.sb_output_dir)

    def test_set_logger(self):
        """Test log file exists."""
        expected_log_file = Path(self.runner._sb_output_dir) / 'sb-run.log'
        self.assertTrue(expected_log_file.is_file())

    def test_get_mode_command(self):
        """Test __get_mode_command."""
        test_cases = [
            {
                'benchmark_name': 'foo',
                'mode': {
                    'name': 'non_exist',
                },
                'expected_command':
                f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo',
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'local',
                    'proc_num': 1,
                    'proc_rank': 0,
                    'prefix': '',
                },
                'expected_command':
                f'PROC_RANK=0 sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo',
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'local',
                    'proc_num': 8,
                    'proc_rank': 6,
                    'prefix': 'CUDA_VISIBLE_DEVICES={proc_rank} numactl -c $(({proc_rank}/2))'
                },
                'expected_command': (
                    'PROC_RANK=6 CUDA_VISIBLE_DEVICES=6 numactl -c $((6/2)) '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo'
                ),
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'local',
                    'proc_num': 16,
                    'proc_rank': 1,
                    'prefix': 'RANK={proc_rank} NUM={proc_num}'
                },
                'expected_command': (
                    'PROC_RANK=1 RANK=1 NUM=16 '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo'
                ),
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'torch.distributed',
                    'proc_num': 1,
                    'node_num': 'all',
                },
                'expected_command': (
                    'python3 -m torch.distributed.launch '
                    '--use_env --no_python --nproc_per_node=1 '
                    '--nnodes=$NNODES --node_rank=$NODE_RANK '
                    '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo '
                    'superbench.benchmarks.foo.parameters.distributed_impl=ddp '
                    'superbench.benchmarks.foo.parameters.distributed_backend=nccl'
                ),
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'torch.distributed',
                    'proc_num': 8,
                    'node_num': 1,
                },
                'expected_command': (
                    'python3 -m torch.distributed.launch '
                    '--use_env --no_python --nproc_per_node=8 '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo '
                    'superbench.benchmarks.foo.parameters.distributed_impl=ddp '
                    'superbench.benchmarks.foo.parameters.distributed_backend=nccl'
                ),
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'mpi',
                    'proc_num': 8,
                    'proc_rank': 1,
                    'mca': {},
                    'env': {
                        'PATH': None,
                        'LD_LIBRARY_PATH': None,
                    },
                },
                'expected_command': (
                    'mpirun -tag-output -allow-run-as-root -hostfile hostfile -map-by ppr:8:node -bind-to numa '
                    ' -x PATH -x LD_LIBRARY_PATH '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo'
                ),
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'mpi',
                    'proc_num': 8,
                    'proc_rank': 2,
                    'mca': {
                        'coll_hcoll_enable': 0,
                    },
                    'env': {
                        'SB_MICRO_PATH': '/sb',
                        'FOO': 'BAR',
                    },
                },
                'expected_command': (
                    'mpirun -tag-output -allow-run-as-root -hostfile hostfile -map-by ppr:8:node -bind-to numa '
                    '-mca coll_hcoll_enable 0 -x SB_MICRO_PATH=/sb -x FOO=BAR '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo'
                ),
            },
        ]
        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                self.assertEqual(
                    self.runner._SuperBenchRunner__get_mode_command(
                        test_case['benchmark_name'], OmegaConf.create(test_case['mode'])
                    ), test_case['expected_command']
                )

    def test_run_empty_benchmarks(self):
        """Test run empty benchmarks, nothing should happen."""
        self.runner._sb_enabled_benchmarks = []
        self.runner.run()

    @mock.patch('superbench.runner.ansible.AnsibleClient.run')
    def test_run_default_benchmarks(self, mock_ansible_client_run):
        """Test run default benchmarks, mock AnsibleClient.run function.

        Args:
            mock_ansible_client_run (function): Mocked AnsibleClient.run function.
        """
        mock_ansible_client_run.return_value = 0
        self.runner.run()

    def test_merge_all_metrics(self):
        """Test __merge_all_metrics."""
        result_summary = json.loads(
            '{"kernel-launch": {"overhead_event": [[0.00583], [0.00545], [0.00581], [0.00572], [0.00559], [0.00591], '
            '[0.00562], [0.00586]], "overhead_wall": [[0.01018], [0.01039], [0.01067], [0.01079], [0.00978], '
            '[0.01085], [0.01036], [0.01033]]}, "resnet_models/pytorch-resnet50": {"steptime_train_float32": '
            '[[252.03], [250.53], [253.75], [250.61], [252.86], [252.58], [251.15], [252.83]], '
            '"throughput_train_float32": [[764.57], [767.83], [762.19], [767.31], [763.41], [764.31], [766.43], '
            '[763.38]], "steptime_train_float16": [[198.36], [196.85], [200.55], [198.07], [199.41], [199.20], '
            '[199.07], [199.34]], "throughput_train_float16": [[972.64], [977.31], [969.58], [974.33], [972.87], '
            '[972.73], [972.46], [972.46]]}, "resnet_models/pytorch-resnet101": {"steptime_train_float32": [[385.53], '
            '[384.05], [386.98], [385.12], [385.47], [385.81], [384.90], [386.65]], "throughput_train_float32": '
            '[[499.39], [500.69], [498.57], [499.83], [499.51], [499.27], [499.94], [498.65]], '
            '"steptime_train_float16": [[307.49], [307.13], [310.31], [307.64], [308.68], [309.61], [307.71], '
            '[309.95]], "throughput_train_float16": [[627.21], [627.34], [624.85], [626.76], [626.26], [625.12], '
            '[626.92], [625.02]]}, "pytorch-sharding-matmul": {"allreduce": [[10.56, 10.66], [10.87, 10.32], '
            '[10.56, 10.45], [10.56, 10.60], [10.56, 10.45], [10.56, 10.38], [10.56, 10.33], [10.56, 10.69]], '
            '"allgather": [[10.08, 10.10], [10.08, 10.16], [10.08, 10.06], [10.56, 10.04], [10.08, 10.05], '
            '[10.08, 10.09], [10.08, 10.08], [10.08, 10.06]]}}'
        )
        reduce_ops = json.loads(
            '{"kernel-launch/overhead_event": null, "kernel-launch/overhead_wall": null, '
            '"resnet_models/pytorch-resnet50/steptime_train_float32": "max", '
            '"resnet_models/pytorch-resnet50/throughput_train_float32": "min", '
            '"resnet_models/pytorch-resnet50/steptime_train_float16": "max", '
            '"resnet_models/pytorch-resnet50/throughput_train_float16": "min", '
            '"resnet_models/pytorch-resnet101/steptime_train_float32": "max", '
            '"resnet_models/pytorch-resnet101/throughput_train_float32": "min", '
            '"resnet_models/pytorch-resnet101/steptime_train_float16": "max", '
            '"resnet_models/pytorch-resnet101/throughput_train_float16": "min", '
            '"pytorch-sharding-matmul/allreduce": "max", "pytorch-sharding-matmul/allgather": "max"}'
        )

        expected = json.loads(
            '{"kernel-launch/overhead_event:0": 0.00583, "kernel-launch/overhead_event:1": 0.00545, '
            '"kernel-launch/overhead_event:2": 0.00581, "kernel-launch/overhead_event:3": 0.00572, '
            '"kernel-launch/overhead_event:4": 0.00559, "kernel-launch/overhead_event:5": 0.00591, '
            '"kernel-launch/overhead_event:6": 0.00562, "kernel-launch/overhead_event:7": 0.00586, '
            '"kernel-launch/overhead_wall:0": 0.01018, "kernel-launch/overhead_wall:1": 0.01039, '
            '"kernel-launch/overhead_wall:2": 0.01067, "kernel-launch/overhead_wall:3": 0.01079, '
            '"kernel-launch/overhead_wall:4": 0.00978, "kernel-launch/overhead_wall:5": 0.01085, '
            '"kernel-launch/overhead_wall:6": 0.01036, "kernel-launch/overhead_wall:7": 0.01033, '
            '"resnet_models/pytorch-resnet50/steptime_train_float32": 253.75, '
            '"resnet_models/pytorch-resnet50/throughput_train_float32": 762.19, '
            '"resnet_models/pytorch-resnet50/steptime_train_float16": 200.55, '
            '"resnet_models/pytorch-resnet50/throughput_train_float16": 969.58, '
            '"resnet_models/pytorch-resnet101/steptime_train_float32": 386.98, '
            '"resnet_models/pytorch-resnet101/throughput_train_float32": 498.57, '
            '"resnet_models/pytorch-resnet101/steptime_train_float16": 310.31, '
            '"resnet_models/pytorch-resnet101/throughput_train_float16": 624.85, '
            '"pytorch-sharding-matmul/0/allreduce": 10.87, "pytorch-sharding-matmul/1/allreduce": 10.69, '
            '"pytorch-sharding-matmul/0/allgather": 10.56, "pytorch-sharding-matmul/1/allgather": 10.16}'
        )
        self.assertEqual(self.runner._SuperBenchRunner__merge_all_metrics(result_summary, reduce_ops), expected)
