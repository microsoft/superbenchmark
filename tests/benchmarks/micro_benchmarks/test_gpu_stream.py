# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the small GPU STREAM preset/fallback interface."""

import json
from pathlib import Path
import unittest
from unittest import mock

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import gpu_stream


@decorator.cuda_test
class GpuStreamBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test explicit presets, native commands, and canonical result parsing."""

    @classmethod
    def setUpClass(cls):
        """Create the existing mock binary environment."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/gpu_stream'])

    def benchmark(self, parameters='', gpu_name=None):
        """Create a benchmark without querying hardware."""
        if gpu_name is not None:
            parameters += ' --gpu_name {}'.format(gpu_name)
        benchmark_class, _ = BenchmarkRegistry._BenchmarkRegistry__select_benchmark('gpu-stream', Platform.CUDA)
        benchmark = benchmark_class('gpu-stream', parameters=parameters)
        self.assertTrue(benchmark._preprocess())
        return benchmark

    @staticmethod
    def output(tags, bandwidth=100):
        """Build synthetic canonical results, not recorded hardware measurements."""
        return '\n'.join(['STREAM Benchmark', 'GPU_STREAM_CONFIG ignored raw launch details'] +
                         ['{}\t{}\t25'.format(tag, bandwidth) for tag in sorted(tags)])

    def test_unknown_chip_with_explicit_size_uses_native_sweep(self):
        """Unknown chips sweep at the requested size without tuning-mode flags."""
        benchmark = self.benchmark('--size 1048576 --num_warm_up 5 --num_loops 10 --check_data')
        self.assertEqual(len(benchmark._commands), 1)
        command = benchmark._commands[0]
        for value in ['--size 1048576', '--num_warm_up 5', '--num_loops 10', '--data_type both', '--check_data']:
            self.assertIn(value, command)
        self.assertNotIn('--block_size', command)
        self.assertNotIn('--mode', command)
        self.assertEqual(len(benchmark._command_targets[0]), 8)

    def test_unknown_chip_falls_back_to_fixed_sweep(self):
        """Future chip names do not inherit another chip's preset."""
        benchmark = self.benchmark(gpu_name='future_gpu')
        self.assertEqual(len(benchmark._commands), 1)
        self.assertIn('--size 8589934592', benchmark._commands[0])
        self.assertNotIn('--block_size', benchmark._commands[0])

    def test_omitted_gpu_name_autotunes_without_loading_presets(self):
        """The normal YAML without a GPU name explicitly takes the sweep path."""
        with mock.patch.object(gpu_stream.json, 'load', side_effect=AssertionError('unexpected preset read')):
            benchmark = self.benchmark()
        self.assertEqual(len(benchmark._commands), 1)
        self.assertNotIn('--block_size', benchmark._commands[0])

    def test_fixed_sweep_excludes_64_threads(self):
        """Keep the reduced four-block by six-unroll fallback space."""
        self.assertEqual(gpu_stream._BLOCKS, [128, 256, 512, 1024])
        self.assertEqual(len(gpu_stream._BLOCKS) * len(gpu_stream._UNROLLS), 24)

    def test_named_chip_uses_only_config_parameters(self):
        """A chip-name entry produces one configured command per workload."""
        benchmark = self.benchmark(gpu_name='h100')
        self.assertEqual(len(benchmark._commands), 8)
        for command in benchmark._commands:
            self.assertIn('--size 4294967296', command)
            self.assertIn('--block_size', command)
            self.assertIn('--unroll', command)
            self.assertNotIn('--grid', command)
            self.assertNotIn('--device_info', command)

    def test_explicit_size_keeps_chip_launch_presets(self):
        """A YAML size override changes every buffer without disabling presets."""
        benchmark = self.benchmark('--size 1048576', gpu_name='h100')
        self.assertEqual(len(benchmark._commands), 8)
        self.assertTrue(all('--size 1048576' in command and '--block_size' in command
                            and '--unroll' in command for command in benchmark._commands))

    def test_legacy_fp64_parameters_remain_valid(self):
        """Preserve the supplied fp64 configuration and use its explicit buffer size."""
        benchmark = self.benchmark(
            '--num_warm_up 10 --num_loops 40 --size 4294967296 --data_type double', gpu_name='vr'
        )
        self.assertEqual(len(benchmark._commands), 4)
        for command in benchmark._commands:
            for value in ['--size 4294967296', '--num_warm_up 10', '--num_loops 40',
                          '--data_type double', '--block_size', '--unroll']:
                self.assertIn(value, command)

    def test_explicit_vr_key_is_supported(self):
        """VR selection is a user-provided key, not a driver-name heuristic."""
        benchmark = self.benchmark(gpu_name='vr')
        self.assertEqual(len(benchmark._commands), 8)
        self.assertTrue(all('--block_size' in command for command in benchmark._commands))

    def test_all_named_gpu_defaults(self):
        """Resolve the three supported JSON keys without any device-name lookup."""
        for gpu_name, size in [('h100', 4294967296), ('gb300', 8589934592), ('vr', 4294967296)]:
            with self.subTest(gpu_name=gpu_name):
                benchmark = self.benchmark(gpu_name=gpu_name)
                self.assertEqual(len(benchmark._commands), 8)
                self.assertTrue(all('--size {}'.format(size) in command for command in benchmark._commands))

    def test_partial_chip_config_sweeps_missing_operations(self):
        """A new profile can be contributed incrementally."""
        profiles = {'example_gpu': {'buffer_size': 1048576,
                    'kernels': {'STREAM_COPY_double': {'block_size': 128, 'unroll': 2}}}}
        with mock.patch.object(gpu_stream.json, 'load', return_value=profiles):
            benchmark = self.benchmark('--data_type double', gpu_name='example_gpu')
        self.assertEqual(len(benchmark._commands), 4)
        self.assertEqual(sum('--block_size' in command for command in benchmark._commands), 1)
        self.assertTrue(all('--size 1048576' in command for command in benchmark._commands))

    def test_profile_file_contains_no_measurement_or_validation_data(self):
        """The public data file contains chip names and launch parameters only."""
        profiles = json.loads(Path(gpu_stream.__file__).with_name('gpu_stream_profiles.json').read_text())
        self.assertTrue(profiles)
        self.assertTrue({'h100', 'gb300', 'vr'}.issubset(profiles))
        for chip, profile in profiles.items():
            self.assertIsInstance(chip, str)
            self.assertEqual(set(profile), {'buffer_size', 'kernels'})
            self.assertIs(type(profile['buffer_size']), int)
            for tag, config in profile['kernels'].items():
                self.assertEqual(set(config), {'block_size', 'unroll'})
                gpu_stream.GpuStreamBenchmark._check_config(tag, config)

    def test_bad_profile_is_an_error_not_a_fallback(self):
        """Invalid matched configuration must not silently become a successful sweep."""
        for config in [
            {'block_size': True, 'unroll': 2},
            {'block_size': 64, 'unroll': 2},
            {'block_size': 96, 'unroll': 2},
            {'block_size': 128, 'unroll': 5},
            {'block_size': 128, 'unroll': 2, 'bandwidth_gbps': 123},
        ]:
            with self.subTest(config=config):
                benchmark_class, _ = BenchmarkRegistry._BenchmarkRegistry__select_benchmark('gpu-stream', Platform.CUDA)
                benchmark = benchmark_class('gpu-stream', parameters='--gpu_name example_gpu')
                with mock.patch.object(gpu_stream.json, 'load',
                                       return_value={'example_gpu': {'buffer_size': 1048576,
                                                     'kernels': {'STREAM_COPY_double': config}}}):
                    self.assertFalse(benchmark._preprocess())
                self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)
                self.assertEqual(benchmark._commands, [])

    def test_invalid_chip_buffer_size_is_reported(self):
        """A shared default must support both scalar widths."""
        for size in [0, 3, True, 2**64]:
            benchmark_class, _ = BenchmarkRegistry._BenchmarkRegistry__select_benchmark('gpu-stream', Platform.CUDA)
            benchmark = benchmark_class('gpu-stream', parameters='--gpu_name example_gpu')
            with mock.patch.object(gpu_stream.json, 'load',
                                   return_value={'example_gpu': {'buffer_size': size, 'kernels': {}}}):
                self.assertFalse(benchmark._preprocess())
            self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)

    def test_gpu_name_is_case_insensitive(self):
        """Match supplied names against the lowercase JSON keys."""
        benchmark = self.benchmark(gpu_name='H100')
        self.assertEqual(benchmark._args.gpu_name, 'h100')
        self.assertEqual(len(benchmark._commands), 8)

    def test_empty_gpu_name_is_rejected(self):
        """Do not silently treat an empty explicit name as a preset."""
        benchmark_class, _ = BenchmarkRegistry._BenchmarkRegistry__select_benchmark('gpu-stream', Platform.CUDA)
        benchmark = benchmark_class('gpu-stream', parameters='--gpu_name \"\"')
        self.assertFalse(benchmark._preprocess())
        self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)

    def test_canonical_results_and_repetitions(self):
        """Publish one fresh value per workload per framework repetition."""
        benchmark = self.benchmark('--size 1048576')
        tags = benchmark._command_targets[0]
        self.assertTrue(benchmark._process_raw_result(0, self.output(tags, 200)))
        self.assertTrue(benchmark._process_raw_result(0, self.output(tags, 100)))
        self.assertEqual(benchmark.result['STREAM_ADD_double_bw'], [200, 100])
        self.assertEqual(len(benchmark.result), benchmark.default_metric_count + 16)
        self.assertTrue(all('_buffer_' not in key and '_block_' not in key for key in benchmark.result))

    def test_single_float_and_scalar_tail_arguments(self):
        """Keep the original single-datatype use case and scalar-aligned sizes."""
        benchmark = self.benchmark('--size 68 --data_type float')
        self.assertIn('--data_type float', benchmark._commands[0])
        self.assertEqual(len(benchmark._command_targets[0]), 4)

    def test_invalid_result_never_returns_bandwidth(self):
        """Reject missing, duplicate, malformed, and nonfinite canonical rows."""
        for transform in [
            lambda output: '',
            lambda output: output + '\nSTREAM_COPY_double 300 25',
            lambda output: output.replace('\t100\t25', '\tnan\t25'),
            lambda output: output.replace('\t100\t25', '\tinf\t25'),
            lambda output: output.replace('\t100\t25', '\t-1\t25'),
        ]:
            benchmark = self.benchmark('--size 1048576 --data_type double')
            self.assertFalse(benchmark._process_raw_result(0, transform(self.output(benchmark._command_targets[0]))))
            self.assertEqual(benchmark.return_code, ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            self.assertNotIn('STREAM_COPY_double_bw', benchmark.result)

    def test_invalid_size_and_loop_counts(self):
        """Retain the normal native input constraints."""
        benchmark_class, _ = BenchmarkRegistry._BenchmarkRegistry__select_benchmark('gpu-stream', Platform.CUDA)
        for parameters in ['--size 0', '--size 12', '--num_loops 0', '--num_warm_up -1']:
            benchmark = benchmark_class('gpu-stream', parameters=parameters)
            self.assertFalse(benchmark._preprocess())
            self.assertEqual(benchmark.return_code, ReturnCode.INVALID_ARGUMENT)
