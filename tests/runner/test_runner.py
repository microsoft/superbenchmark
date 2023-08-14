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
        test_config_file = Path(__file__).parent / '../../tests/data/test.yaml'
        with test_config_file.open() as fp:
            self.test_config = OmegaConf.create(yaml.load(fp, Loader=yaml.SafeLoader))
        self.sb_output_dir = tempfile.mkdtemp()

        self.runner = SuperBenchRunner(
            self.test_config,
            OmegaConf.create({}),
            OmegaConf.create({}),
            self.sb_output_dir,
        )

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        shutil.rmtree(self.sb_output_dir)

    def test_set_logger(self):
        """Test log file exists."""
        expected_log_file = Path(self.runner._sb_output_dir) / 'sb-run.log'
        self.assertTrue(expected_log_file.is_file())

    def test_get_failure_count(self):
        """Test get_failure_count."""
        self.assertEqual(0, self.runner.get_failure_count())

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
                    'prefix': 'CUDA_VISIBLE_DEVICES={proc_rank} numactl -N $(({proc_rank}/2))'
                },
                'expected_command': (
                    'PROC_RANK=6 CUDA_VISIBLE_DEVICES=6 numactl -N $((6/2)) '
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
                    'torchrun '
                    '--no_python --nproc_per_node=1 '
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
                    'torchrun '
                    '--no_python --nproc_per_node=8 '
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
                        'RANK': '{proc_rank}',
                        'NUM': '{proc_num}',
                    },
                },
                'expected_command': (
                    'mpirun -tag-output -allow-run-as-root -hostfile hostfile -map-by ppr:8:node -bind-to numa '
                    '-mca coll_hcoll_enable 0 -x SB_MICRO_PATH=/sb -x FOO=BAR -x RANK=2 -x NUM=8 '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo'
                ),
            },
            {
                'benchmark_name':
                'foo',
                'mode': {
                    'name': 'mpi',
                    'node_num': 1,
                    'proc_num': 8,
                    'proc_rank': 2,
                    'mca': {
                        'coll_hcoll_enable': 0,
                    },
                    'env': {
                        'SB_MICRO_PATH': '/sb',
                        'FOO': 'BAR',
                        'RANK': '{proc_rank}',
                        'NUM': '{proc_num}',
                    },
                },
                'expected_command': (
                    'mpirun -tag-output -allow-run-as-root -host localhost:8 -bind-to numa '
                    '-mca coll_hcoll_enable 0 -x SB_MICRO_PATH=/sb -x FOO=BAR -x RANK=2 -x NUM=8 '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo'
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
                    'pattern': {
                        'type': 'all-nodes',
                    },
                    'env': {
                        'PATH': None,
                        'LD_LIBRARY_PATH': None,
                    },
                },
                'expected_command': (
                    'mpirun -tag-output -allow-run-as-root -host node0:8,node1:8 -bind-to numa '
                    ' -x PATH -x LD_LIBRARY_PATH '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo'
                ),
            },
        ]

        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                mode = OmegaConf.create(test_case['mode'])
                if 'pattern' in test_case['mode']:
                    mode.update({'host_list': ['node0', 'node1']})
                self.assertEqual(
                    self.runner._SuperBenchRunner__get_mode_command(
                        test_case['benchmark_name'],
                        mode,
                    ), test_case['expected_command']
                )

                test_case['timeout'] = 10
                timeout_str = 'timeout {} '.format(test_case['timeout'])
                index = test_case['expected_command'].find('sb exec')
                expected_command = test_case['expected_command'][:index] + timeout_str + test_case['expected_command'][
                    index:]
                mode = OmegaConf.create(test_case['mode'])
                if 'pattern' in test_case['mode']:
                    mode.update({'host_list': ['node0', 'node1']})
                self.assertEqual(
                    self.runner._SuperBenchRunner__get_mode_command(
                        test_case['benchmark_name'],
                        mode,
                        test_case['timeout'],
                    ), expected_command
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

    def test_merge_benchmark_metrics(self):
        """Test __merge_benchmark_metrics."""
        result_summary = json.loads(
            '{"kernel-launch": {"overhead_event": [[0.00583], [0.00545], [0.00581], [0.00572], [0.00559], [0.00591], '
            '[0.00562], [0.00586]], "overhead_wall": [[0.01018], [0.01039], [0.01067], [0.01079], [0.00978], '
            '[0.01085], [0.01036], [0.01033]]}, "resnet_models/pytorch-resnet50": {"steptime_train_float32": '
            '[[252.03]], "throughput_train_float32": [[764.57]], "steptime_train_float16": [[198.36]], '
            '"throughput_train_float16": [[972.64]]}, "resnet_models/pytorch-resnet101": {"steptime_train_float32": '
            '[[385.53]], "throughput_train_float32": [[499.39]], "steptime_train_float16": [[307.49]], '
            '"throughput_train_float16": [[627.21]]}, "pytorch-sharding-matmul": {"allreduce": [[10.56, 10.66], '
            '[10.87, 10.32], [10.56, 10.45], [10.56, 10.60], [10.56, 10.45], [10.56, 10.38], [10.56, 10.33], '
            '[10.56, 10.69]], "allgather": [[10.08, 10.10], [10.08, 10.16], [10.08, 10.06], [10.56, 10.04], '
            '[10.08, 10.05], [10.08, 10.09], [10.08, 10.08], [10.08, 10.06]]}}'
        )
        reduce_ops = json.loads(
            '{"kernel-launch/overhead_event": null, "kernel-launch/overhead_wall": null, '
            '"resnet_models/pytorch-resnet50/steptime_train_float32": null, '
            '"resnet_models/pytorch-resnet50/throughput_train_float32": null, '
            '"resnet_models/pytorch-resnet50/steptime_train_float16": null, '
            '"resnet_models/pytorch-resnet50/throughput_train_float16": null, '
            '"resnet_models/pytorch-resnet101/steptime_train_float32": null, '
            '"resnet_models/pytorch-resnet101/throughput_train_float32": null, '
            '"resnet_models/pytorch-resnet101/steptime_train_float16": null, '
            '"resnet_models/pytorch-resnet101/throughput_train_float16": null, '
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
            '"resnet_models/pytorch-resnet50/steptime_train_float32": 252.03, '
            '"resnet_models/pytorch-resnet50/throughput_train_float32": 764.57, '
            '"resnet_models/pytorch-resnet50/steptime_train_float16": 198.36, '
            '"resnet_models/pytorch-resnet50/throughput_train_float16": 972.64, '
            '"resnet_models/pytorch-resnet101/steptime_train_float32": 385.53, '
            '"resnet_models/pytorch-resnet101/throughput_train_float32": 499.39, '
            '"resnet_models/pytorch-resnet101/steptime_train_float16": 307.49, '
            '"resnet_models/pytorch-resnet101/throughput_train_float16": 627.21, '
            '"pytorch-sharding-matmul/0/allreduce": 10.87, "pytorch-sharding-matmul/1/allreduce": 10.69, '
            '"pytorch-sharding-matmul/0/allgather": 10.56, "pytorch-sharding-matmul/1/allgather": 10.16}'
        )
        self.assertEqual(self.runner._SuperBenchRunner__merge_benchmark_metrics(result_summary, reduce_ops), expected)

    def test_merge_monitor_metrics(self):
        """Test __merge_monitor_metrics."""
        path = Path('tests/data/monitor/')
        expected = {
            'monitor/gpu_temperature:0': 50,
            'monitor/gpu_temperature:1': 27,
            'monitor/gpu_temperature:2': 24,
            'monitor/gpu_temperature:3': 26,
            'monitor/gpu_temperature:4': 25,
            'monitor/gpu_temperature:5': 25,
            'monitor/gpu_temperature:6': 23,
            'monitor/gpu_temperature:7': 26,
            'monitor/gpu_power_limit:0': 250,
            'monitor/gpu_power_limit:1': 200,
            'monitor/gpu_power_limit:2': 250,
            'monitor/gpu_power_limit:3': 250,
            'monitor/gpu_power_limit:4': 250,
            'monitor/gpu_power_limit:5': 250,
            'monitor/gpu_power_limit:6': 250,
            'monitor/gpu_power_limit:7': 250,
            'monitor/gpu_corrected_ecc:0': 12,
            'monitor/gpu_corrected_ecc:1': 0,
            'monitor/gpu_corrected_ecc:2': 0,
            'monitor/gpu_corrected_ecc:3': 0,
            'monitor/gpu_corrected_ecc:4': 0,
            'monitor/gpu_corrected_ecc:5': 0,
            'monitor/gpu_corrected_ecc:6': 0,
            'monitor/gpu_corrected_ecc:7': 0,
            'monitor/gpu_uncorrected_ecc:0': 0,
            'monitor/gpu_uncorrected_ecc:1': 0,
            'monitor/gpu_uncorrected_ecc:2': 0,
            'monitor/gpu_uncorrected_ecc:3': 0,
            'monitor/gpu_uncorrected_ecc:4': 0,
            'monitor/gpu_uncorrected_ecc:5': 0,
            'monitor/gpu_uncorrected_ecc:6': 0,
            'monitor/gpu_uncorrected_ecc:7': 0
        }
        self.assertEqual(self.runner._SuperBenchRunner__merge_monitor_metrics(path), expected)

    def test_generate_metric_name(self):
        """Test __generate_metric_name."""
        """(self, benchmark_name, metric, rank_count, run_count, curr_rank, curr_run):"""
        test_cases = [
            {
                'benchmark_name': 'kernel-launch',
                'metric': 'overhead_event',
                'rank_count': 8,
                'run_count': 2,
                'curr_rank': 0,
                'curr_run': 0,
                'expected': 'kernel-launch/0/overhead_event:0',
            },
            {
                'benchmark_name': 'kernel-launch',
                'metric': 'overhead_event',
                'rank_count': 8,
                'run_count': 2,
                'curr_rank': 2,
                'curr_run': 1,
                'expected': 'kernel-launch/1/overhead_event:2',
            },
            {
                'benchmark_name': 'kernel-launch',
                'metric': 'overhead_event',
                'rank_count': 1,
                'run_count': 1,
                'curr_rank': 0,
                'curr_run': 0,
                'expected': 'kernel-launch/overhead_event',
            },
            {
                'benchmark_name': 'resnet_models/pytorch-resnet50',
                'metric': 'fp32_train_step_time',
                'rank_count': 1,
                'run_count': 2,
                'curr_rank': 0,
                'curr_run': 1,
                'expected': 'resnet_models/pytorch-resnet50/1/fp32_train_step_time',
            },
            {
                'benchmark_name': 'resnet_models/pytorch-resnet50',
                'metric': 'fp32_train_step_time',
                'rank_count': 1,
                'run_count': 1,
                'curr_rank': 0,
                'curr_run': 0,
                'expected': 'resnet_models/pytorch-resnet50/fp32_train_step_time',
            },
        ]

        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                self.assertEqual(
                    self.runner._SuperBenchRunner__generate_metric_name(
                        test_case['benchmark_name'], test_case['metric'], test_case['rank_count'],
                        test_case['run_count'], test_case['curr_rank'], test_case['curr_run']
                    ), test_case['expected']
                )
