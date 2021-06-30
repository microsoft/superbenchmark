# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner test."""

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
                'benchmark_name': 'foo',
                'mode': {
                    'name': 'local',
                    'proc_num': 1,
                    'prefix': '',
                },
                'expected_command':
                f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo',
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
                    'CUDA_VISIBLE_DEVICES=6 numactl -c $((6/2)) '
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
                'expected_command':
                f'RANK=1 NUM=16 sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo',
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
                    '--nnodes=1 --node_rank=$NODE_RANK '
                    '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                    f'sb exec --output-dir {self.sb_output_dir} -c sb.config.yaml -C superbench.enable=foo '
                    'superbench.benchmarks.foo.parameters.distributed_impl=ddp '
                    'superbench.benchmarks.foo.parameters.distributed_backend=nccl'
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
