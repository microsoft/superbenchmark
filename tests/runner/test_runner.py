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
        self.assertEqual(self.runner._SuperBenchRunner__get_mode_command('non_exist', 'sb exec'), 'sb exec')
        self.assertEqual(
            self.runner._SuperBenchRunner__get_mode_command('torch.distributed', 'sb exec'), (
                'python3 -m torch.distributed.launch '
                '--use_env --no_python --nproc_per_node=8 '
                '--nnodes=$NNODES --node_rank=$NODE_RANK '
                '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                'sb exec'
            )
        )

    def test_run(self):
        """Test run."""
        self.runner._sb_enabled_benchmarks = []
        self.runner.run()
