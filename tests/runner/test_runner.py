# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner test."""

import unittest
import shutil
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from superbench.runner import SuperBenchRunner


class RunnerTestCase(unittest.TestCase):
    """A class for runner test cases."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        default_config_file = Path(__file__).parent / '../../superbench/config/default.yaml'
        self.default_config = OmegaConf.load(str(default_config_file))
        self.output_dir = tempfile.mkdtemp()

        self.runner = SuperBenchRunner(self.default_config, None, None, self.output_dir)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        shutil.rmtree(self.output_dir)

    def test_set_logger(self):
        """Test log file exists."""
        expected_log_file = Path(self.runner._output_dir) / 'sb-run.log'
        self.assertTrue(expected_log_file.is_file())

    def test_get_mode_command(self):
        """Test __get_mode_command."""
        test_cases = [
            {
                'mode': {
                    'name': 'non_exist',
                },
                'exec_command': 'sb exec',
                'expected_command': 'sb exec',
            },
            {
                'mode': {
                    'name': 'local',
                },
                'exec_command': 'sb exec',
                'expected_command': 'sb exec',
            },
            {
                'mode': {
                    'name': 'local',
                    'proc_num': 8,
                    'proc_rank': 6,
                    'prefix': 'CUDA_VISIBLE_DEVICES={proc_rank} numactl -c $(({proc_rank}/2))'
                },
                'exec_command': 'sb exec',
                'expected_command': 'CUDA_VISIBLE_DEVICES=6 numactl -c $((6/2)) sb exec',
            },
            {
                'mode': {
                    'name': 'local',
                    'proc_num': 16,
                    'proc_rank': 1,
                    'prefix': 'RANK={proc_rank} NUM={proc_num}'
                },
                'exec_command': 'sb exec',
                'expected_command': 'RANK=1 NUM=16 sb exec',
            },
            {
                'mode': {
                    'name': 'torch.distributed',
                },
                'exec_command':
                'sb exec',
                'expected_command': (
                    'python3 -m torch.distributed.launch '
                    '--use_env --no_python --nproc_per_node=8 '
                    '--nnodes=$NNODES --node_rank=$NODE_RANK '
                    '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                    'sb exec'
                ),
            },
            {
                'mode': {
                    'name': 'torch.distributed',
                    'proc_num': 1,
                    'node_num': 'all',
                },
                'exec_command':
                'sb exec',
                'expected_command': (
                    'python3 -m torch.distributed.launch '
                    '--use_env --no_python --nproc_per_node=1 '
                    '--nnodes=$NNODES --node_rank=$NODE_RANK '
                    '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                    'sb exec'
                ),
            },
            {
                'mode': {
                    'name': 'torch.distributed',
                    'proc_num': 8,
                    'node_num': 1,
                },
                'exec_command':
                'sb exec',
                'expected_command': (
                    'python3 -m torch.distributed.launch '
                    '--use_env --no_python --nproc_per_node=8 '
                    '--nnodes=1 --node_rank=$NODE_RANK '
                    '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                    'sb exec'
                ),
            },
        ]
        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                self.assertEqual(
                    self.runner._SuperBenchRunner__get_mode_command(
                        OmegaConf.create(test_case['mode']), test_case['exec_command']
                    ), test_case['expected_command']
                )

    def test_run(self):
        """Test run."""
        self.runner._sb_enabled_benchmarks = []
        self.runner.run()
