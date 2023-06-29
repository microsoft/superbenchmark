# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for tensorrt-inference benchmark."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.result import BenchmarkResult


class TensorRTInferenceBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
    """Class for tensorrt-inferencee benchmark test cases."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'tensorrt-inference'
        cls._model_path = Path(cls._tmp_dir) / 'hub' / 'onnx'
        cls.createMockEnvs(cls, {
            'TORCH_HOME': cls._tmp_dir,
            'SB_MICRO_PATH': cls._tmp_dir,
        })
        cls.createMockFiles(cls, ['bin/trtexec'])

    def test_tensorrt_inference_cls(self):
        """Test tensorrt-inference benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.CUDA:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    @decorator.cuda_test
    @decorator.pytorch_test
    def test_tensorrt_inference_params(self):
        """Test tensorrt-inference benchmark preprocess with different parameters."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)

        test_cases = [
            {
                'precision': 'fp32',
            },
            {
                'pytorch_models': ['resnet50', 'mnasnet0_5'],
                'precision': 'fp16',
            },
            {
                'pytorch_models': ['resnet50'],
                'batch_size': 4,
            },
            {
                'pytorch_models': ['lstm', 'bert-base', 'gpt2-small'],
                'batch_size': 4,
                'seq_length': 128,
                'iterations': 256,
            },
        ]
        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                parameter_list = []
                if 'pytorch_models' in test_case:
                    parameter_list.append(f'--pytorch_models {" ".join(test_case["pytorch_models"])}')
                if 'precision' in test_case:
                    parameter_list.append(f'--precision {test_case["precision"]}')
                if 'batch_size' in test_case:
                    parameter_list.append(f'--batch_size {test_case["batch_size"]}')
                if 'seq_length' in test_case:
                    parameter_list.append(f'--seq_length {test_case["seq_length"]}')
                if 'iterations' in test_case:
                    parameter_list.append(f'--iterations {test_case["iterations"]}')

                # Check basic information
                benchmark = benchmark_cls(self.benchmark_name, parameters=' '.join(parameter_list))
                self.assertTrue(benchmark)

                # Limit model number
                benchmark._pytorch_models = benchmark._pytorch_models[:1]

                # Preprocess
                ret = benchmark._preprocess()
                self.assertTrue(ret)
                self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)
                self.assertEqual(BenchmarkType.MICRO, benchmark.type)
                self.assertEqual(self.benchmark_name, benchmark.name)

                # Check parameters
                self.assertEqual(
                    test_case.get('pytorch_models', benchmark._pytorch_models),
                    benchmark._args.pytorch_models,
                )
                self.assertEqual(
                    test_case.get('precision', 'int8'),
                    benchmark._args.precision,
                )
                self.assertEqual(
                    test_case.get('batch_size', 32),
                    benchmark._args.batch_size,
                )
                self.assertEqual(
                    test_case.get('iterations', 2048),
                    benchmark._args.iterations,
                )

                # Check models
                for model in benchmark._args.pytorch_models:
                    self.assertTrue((self._model_path / f'{model}.onnx').is_file())

                # Command list should equal to default model number
                self.assertEqual(
                    len(test_case.get('pytorch_models', benchmark._pytorch_models)), len(benchmark._commands)
                )

    @decorator.load_data('tests/data/tensorrt_inference.1.log')
    @decorator.load_data('tests/data/tensorrt_inference.2.log')
    def test_tensorrt_inference_result_parsing(self, test_raw_log_1, test_raw_log_2):
        """Test tensorrt-inference benchmark result parsing."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        benchmark = benchmark_cls(self.benchmark_name, parameters='')
        benchmark._args = SimpleNamespace(pytorch_models=['model_0', 'model_1'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # Positive case 1 - valid raw output
        self.assertTrue(benchmark._process_raw_result(0, test_raw_log_1))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertEqual(6 + benchmark.default_metric_count, len(benchmark.result))
        for tag in ['mean', '99']:
            self.assertEqual(0.5, benchmark.result[f'model_0_gpu_time_{tag}'][0])
            self.assertEqual(0.6, benchmark.result[f'model_0_host_time_{tag}'][0])
            self.assertEqual(1.0, benchmark.result[f'model_0_end_to_end_time_{tag}'][0])

        # Positive case 2 - valid raw output
        self.assertTrue(benchmark._process_raw_result(0, test_raw_log_2))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)
        for tag in ['mean', '99']:
            self.assertEqual(1.5, benchmark.result[f'model_0_gpu_time_{tag}'][1])
            self.assertEqual(2.0, benchmark.result[f'model_0_host_time_{tag}'][1])

        # Negative case - invalid raw output
        self.assertFalse(benchmark._process_raw_result(1, 'Invalid raw output'))
