# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ort-inference benchmark."""

import shutil
from pathlib import Path
from unittest import mock

import torch
import torchvision.models

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Precision, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.ort_inference_performance import ORTInferenceBenchmark


@decorator.cuda_test
@decorator.pytorch_test
@mock.patch('torch.hub.get_dir')
@mock.patch('onnxruntime.InferenceSession.run')
def test_ort_inference_performance(mock_ort_session_run, mock_get_dir):
    """Test ort-inference benchmark."""
    benchmark_name = 'ort-inference'
    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
    assert (benchmark_class)

    mock_get_dir.return_value = '/tmp/superbench/'
    benchmark = benchmark_class(
        benchmark_name,
        parameters='--pytorch_models resnet50 --graph_opt_level 1 --precision float16'
        ' --batch_size 16 --num_warmup 128 --num_steps 512'
    )

    assert (isinstance(benchmark, ORTInferenceBenchmark))
    assert (benchmark._preprocess())

    # Check basic information.
    assert (benchmark.name == 'ort-inference')
    assert (benchmark.type == BenchmarkType.MICRO)
    assert (benchmark._ORTInferenceBenchmark__model_cache_path == Path(torch.hub.get_dir()) / 'checkpoints')
    for model in benchmark._args.pytorch_models:
        assert (hasattr(torchvision.models, model))
        file_name = '{model}.{precision}.onnx'.format(model=model, precision=benchmark._args.precision)
        assert ((benchmark._ORTInferenceBenchmark__model_cache_path / file_name).is_file())

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.pytorch_models == ['resnet50'])
    assert (benchmark._args.graph_opt_level == 1)
    assert (benchmark._args.precision == Precision.FLOAT16)
    assert (benchmark._args.batch_size == 16)
    assert (benchmark._args.num_warmup == 128)
    assert (benchmark._args.num_steps == 512)

    # Check results and metrics.
    assert (benchmark._benchmark())
    shutil.rmtree(benchmark._ORTInferenceBenchmark__model_cache_path)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'int8': 'int8'}
    for model in benchmark._args.pytorch_models:
        if benchmark._args.precision.value in precision_metric:
            precision = precision_metric[benchmark._args.precision.value]
        else:
            precision = benchmark._args.precision.value
        metric = '{}_{}_time'.format(precision, model)
        assert (metric in benchmark.result)
        assert (metric in benchmark.raw_data)
