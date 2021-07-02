# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Executor test."""

import json
import unittest
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import yaml
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
        with default_config_file.open() as fp:
            self.default_config = OmegaConf.create(yaml.load(fp, Loader=yaml.SafeLoader))
        self.sb_output_dir = tempfile.mkdtemp()

        self.executor = SuperBenchExecutor(self.default_config, self.sb_output_dir)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        shutil.rmtree(self.sb_output_dir)

    def test_set_logger(self):
        """Test log file exists."""
        expected_log_file = Path(self.executor._sb_output_dir) / 'sb-exec.log'
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
        test_cases = [
            {
                'parameters': None,
                'expected_args': '',
            },
            {
                'parameters': {
                    'duration': 0,
                    'num_warmup': 16,
                    'num_steps': 128,
                    'batch_size': 16,
                    'precision': ['float32', 'float16'],
                    'model_action': ['train', 'inference'],
                },
                'expected_args': (
                    '--duration 0 --num_warmup 16 --num_steps 128 --batch_size 16 '
                    '--precision float32 float16 --model_action train inference'
                ),
            },
        ]
        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                self.assertEqual(
                    self.executor._SuperBenchExecutor__get_arguments(test_case['parameters']),
                    test_case['expected_args']
                )

    def test_create_benchmark_dir(self):
        """Test __create_benchmark_dir."""
        foo_path = Path(self.sb_output_dir, 'benchmarks', 'foo', 'rank0')
        self.executor._SuperBenchExecutor__create_benchmark_dir('foo')
        self.assertTrue(foo_path.is_dir())
        self.assertFalse(any(foo_path.iterdir()))

        (foo_path / 'bar.txt').touch()
        self.executor._SuperBenchExecutor__create_benchmark_dir('foo')
        self.assertTrue(foo_path.is_dir())
        self.assertFalse(any(foo_path.iterdir()))
        self.assertFalse((foo_path / 'bar.txt').is_file())
        self.assertTrue(foo_path.with_name('rank0.bak1').is_dir())
        self.assertTrue((foo_path.with_name('rank0.bak1') / 'bar.txt').is_file())

        (foo_path / 'bar.json').touch()
        self.executor._SuperBenchExecutor__create_benchmark_dir('foo')
        self.assertTrue(foo_path.is_dir())
        self.assertFalse(any(foo_path.iterdir()))
        self.assertFalse((foo_path / 'bar.json').is_file())
        self.assertTrue(foo_path.with_name('rank0.bak2').is_dir())
        self.assertTrue((foo_path.with_name('rank0.bak2') / 'bar.json').is_file())

    def test_write_benchmark_results(self):
        """Test __write_benchmark_results."""
        foobar_path = Path(self.sb_output_dir, 'benchmarks', 'foobar', 'rank0')
        foobar_results_path = foobar_path / 'results.json'
        self.executor._SuperBenchExecutor__create_benchmark_dir('foobar')
        foobar_results = {
            'sum': 1,
            'avg': 1.1,
        }
        self.executor._SuperBenchExecutor__write_benchmark_results('foobar', foobar_results)
        self.assertTrue(foobar_results_path.is_file())
        with foobar_results_path.open(mode='r') as f:
            self.assertDictEqual(json.load(f), foobar_results)

    def test_exec_empty_benchmarks(self):
        """Test execute empty benchmarks, nothing should happen."""
        self.executor._sb_enabled = []
        self.executor.exec()

    @mock.patch('superbench.executor.SuperBenchExecutor._SuperBenchExecutor__exec_benchmark')
    def test_exec_default_benchmarks(self, mock_exec_benchmark):
        """Test execute default benchmarks, mock exec function.

        Args:
            mock_exec_benchmark (function): Mocked __exec_benchmark function.
        """
        mock_exec_benchmark.return_value = {}
        self.executor.exec()

        self.assertTrue(Path(self.sb_output_dir, 'benchmarks').is_dir())
        for benchmark_name in self.executor._sb_benchmarks:
            self.assertTrue(Path(self.sb_output_dir, 'benchmarks', benchmark_name, 'rank0').is_dir())
            self.assertTrue(Path(self.sb_output_dir, 'benchmarks', benchmark_name, 'rank0', 'results.json').is_file())
