# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gpu-stream benchmark."""

import json
from pathlib import Path
import shlex
import unittest

from omegaconf import OmegaConf

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.executor import SuperBenchExecutor

_TAGS = ['STREAM_COPY_double', 'STREAM_SCALE_double', 'STREAM_ADD_double', 'STREAM_TRIAD_double']


@decorator.cuda_test
class GpuStreamBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test command generation and result parsing with synthetic data."""
    @classmethod
    def setUpClass(cls):
        """Create mock binary environment."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/gpu_stream'])

    def create(self, parameters='', configs=None):
        """Create a benchmark and run preprocess."""
        if configs is not None:
            parameters += ' --kernel_configs {}'.format(shlex.quote(json.dumps(configs)))
        benchmark_class, _ = BenchmarkRegistry._BenchmarkRegistry__select_benchmark('gpu-stream', Platform.CUDA)
        benchmark = benchmark_class('gpu-stream', parameters=parameters)
        return benchmark, benchmark._preprocess()

    @staticmethod
    def output(tags, bandwidth=100):
        """Build synthetic native output."""
        return '\n'.join(['STREAM Benchmark'] + ['{}\t{}\t25'.format(tag, bandwidth) for tag in sorted(tags)])

    def test_native_sweep(self):
        """Without kernel_configs, one native command sweeps all operations."""
        benchmark, ret = self.create('--size 1048576 --num_warm_up 5 --num_loops 10 --check_data')
        self.assertTrue(ret)
        self.assertEqual(len(benchmark._commands), 1)
        for value in [
            '--size 1048576', '--data_type double', '--operation all', '--num_warm_up 5', '--num_loops 10',
            '--check_data'
        ]:
            self.assertIn(value, benchmark._commands[0])
        self.assertNotIn('--block_size', benchmark._commands[0])
        self.assertEqual(benchmark._command_targets, [set(_TAGS)])

    def test_kernel_configs(self):
        """Listed workloads use their launch config; unlisted workloads sweep."""
        configs = {
            'STREAM_COPY_double': {
                'block_size': 256,
                'unroll': 2
            },
            'STREAM_TRIAD_double': {
                'block_size': 512,
                'unroll': 4
            }
        }
        benchmark, ret = self.create('--size 1048576', configs)
        self.assertTrue(ret)
        self.assertEqual(benchmark._command_targets, [{tag} for tag in _TAGS])
        for command, tag in zip(benchmark._commands, _TAGS):
            self.assertIn('--size 1048576', command)
            self.assertIn('--operation {}'.format(tag.split('_')[1].lower()), command)
            if tag in configs:
                self.assertIn('--block_size {block_size} --unroll {unroll}'.format(**configs[tag]), command)
            else:
                self.assertNotIn('--block_size', command)

    def test_kernel_configs_from_yaml(self):
        """Inline JSON in YAML survives executor argument conversion."""
        config = OmegaConf.create(
            """
parameters:
  data_type: double
  kernel_configs: >-
    '{
      "STREAM_COPY_double":  {"block_size": 256, "unroll": 2},
      "STREAM_SCALE_double": {"block_size": 256, "unroll": 2},
      "STREAM_ADD_double":   {"block_size": 256, "unroll": 2},
      "STREAM_TRIAD_double": {"block_size": 256, "unroll": 2}
    }'
"""
        )
        executor = SuperBenchExecutor.__new__(SuperBenchExecutor)
        benchmark, ret = self.create(executor._SuperBenchExecutor__get_arguments(config.parameters))
        self.assertTrue(ret)
        self.assertEqual(len(benchmark._commands), 4)
        self.assertTrue(all('--block_size 256 --unroll 2' in command for command in benchmark._commands))

    def test_invalid_arguments(self):
        """Invalid sizes, loops, and kernel_configs are rejected."""
        cases = ['--size 0', '--size 12', '--num_loops 0', '--num_warm_up -1']
        for value in [
            '{',
            '[]',
            '{"STREAM_COPY_half": {"block_size": 256, "unroll": 2}}',
            '{"STREAM_COPY_float": {"block_size": 256, "unroll": 2}}',
            '{"STREAM_COPY_double": {"block_size": 256}}',
            '{"STREAM_COPY_double": {"block_size": 64, "unroll": 2}}',
            '{"STREAM_COPY_double": {"block_size": 256, "unroll": 5}}',
            '{"STREAM_COPY_double": {"block_size": true, "unroll": 2}}',
        ]:
            cases.append('--kernel_configs {}'.format(shlex.quote(value)))
        for parameters in cases:
            with self.subTest(parameters=parameters):
                benchmark, ret = self.create(parameters)
                self.assertFalse(ret)
                self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)
                self.assertEqual(benchmark._commands, [])

    def test_result_parsing(self):
        """Metrics come from canonical rows; launch details stay in raw output."""
        for dtype, filename in [('float', 'gpu_stream_float.log'), ('double', 'gpu_stream.log')]:
            with self.subTest(dtype=dtype):
                benchmark, ret = self.create('--data_type {}'.format(dtype))
                self.assertTrue(ret)
                output = (Path(__file__).parents[2] / 'data' / filename).read_text()
                self.assertTrue(benchmark._process_raw_result(0, output))
                rows = [line.split() for line in output.splitlines() if line.startswith('STREAM_')]
                self.assertEqual(len(rows), 4)
                for tag, bandwidth, ratio in rows:
                    self.assertEqual(benchmark.result[tag + '_bw'], [float(bandwidth)])
                    self.assertEqual(benchmark.result[tag + '_ratio'], [float(ratio)])
                self.assertEqual(len(benchmark.result), benchmark.default_metric_count + 8)

    def test_invalid_result(self):
        """Missing, duplicate, and non-finite rows fail parsing."""
        valid = self.output(_TAGS)
        for output in [
            '',
            valid + '\nSTREAM_COPY_double\t300\t25',
            valid.replace('\t100\t25', '\tnan\t25'),
            valid.replace('\t100\t25', '\t-1\t25'),
        ]:
            with self.subTest(output=output):
                benchmark, _ = self.create()
                self.assertFalse(benchmark._process_raw_result(0, output))
                self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
                self.assertNotIn('STREAM_COPY_double_bw', benchmark.result)
