# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Executor test."""

import unittest
import shutil
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from superbench.executor import SuperBenchExecutor


class ExecutorTestCase(unittest.TestCase):
    """A class for executor test cases.

    Args:
        unittest.TestCase (unittest.TestCase): TestCase class.
    """
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        default_config_file = Path(__file__).parent / '../../superbench/config/default.yaml'
        self.default_config = OmegaConf.load(str(default_config_file))
        self.output_dir = tempfile.mkdtemp()

        self.executor = SuperBenchExecutor(self.default_config, None, self.output_dir)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        shutil.rmtree(self.output_dir)

    def test_set_logger(self):
        """Test log file exists."""
        expected_log_file = Path(self.executor._output_dir) / 'sb-exec.log'
        self.assertTrue(expected_log_file.is_file())

    def test_get_enabled_benchmarks_enable_none(self):
        """Test enabled benchmarks when superbench.enable is none."""
        expected_enabled_benchmarks = list(self.default_config.superbench.benchmarks.keys())
        self.assertListEqual(self.executor._sb_enabled, expected_enabled_benchmarks)

    def test_get_enabled_benchmarks_enable_str(self):
        """Test enabled benchmarks when superbench.enable is string."""
        self.executor._sb_config.superbench.enable = 'benchmark_alpha'
        expected_enabled_benchmarks = ['benchmark_alpha']
        self.assertListEqual(self.executor._SuperBenchExecutor__get_enabled_benchmarks(), expected_enabled_benchmarks)

    def test_get_enabled_benchmarks_enable_list(self):
        """Test enabled benchmarks when superbench.enable is list."""
        self.executor._sb_config.superbench.enable = ['benchmark_alpha', 'benchmark_beta']
        expected_enabled_benchmarks = ['benchmark_alpha', 'benchmark_beta']
        self.assertListEqual(self.executor._SuperBenchExecutor__get_enabled_benchmarks(), expected_enabled_benchmarks)

    def test_get_platform(self):
        """Test get platform."""
        self.assertEqual(self.executor._SuperBenchExecutor__get_platform().value, 'CUDA')

    def test_get_arguments(self):
        """Test benchmarks arguments."""
        expected_bert_models_args = \
            '--duration 0 --num_warmup 64 --num_steps 2048 --batch_size 32 ' \
            '--precision float32 float16 --model_action train inference'
        self.assertEqual(
            self.executor._SuperBenchExecutor__get_arguments(
                self.default_config.superbench.benchmarks.bert_models.parameters
            ), expected_bert_models_args
        )

    def test_exec_empty_benchmarks(self):
        """Test execute empty benchmarks, nothing should happen."""
        self.executor._sb_enabled = []
        self.executor.exec()
