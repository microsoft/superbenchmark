# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for CNN model benchmarks."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_cnn import PytorchCNN


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_cnn_with_gpu():
    """Test pytorch cnn benchmarks with GPU."""
    run_pytorch_cnn(
        models=['resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19'],
        parameters='--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 \
            --model_action train inference',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp16_train_step_time', 'fp16_train_throughput',
            'fp32_inference_step_time', 'fp32_inference_throughput', 'fp16_inference_step_time',
            'fp16_inference_throughput'
        ]
    )


@decorator.pytorch_test
def test_pytorch_cnn_no_gpu():
    """Test pytorch cnn benchmarks with CPU."""
    run_pytorch_cnn(
        models=['resnet50'],
        parameters='--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 \
                --model_action train inference --precision float32 --no_gpu',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp32_inference_step_time', 'fp32_inference_throughput'
        ]
    )


def run_pytorch_cnn(models=[], parameters='', check_metrics=[]):
    """Run pytorch cnn benchmarks."""
    for model in models:
        context = BenchmarkRegistry.create_benchmark_context(
            model, platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
        )

        assert (BenchmarkRegistry.is_benchmark_context_valid(context))

        benchmark = BenchmarkRegistry.launch_benchmark(context)

        # Check basic information.
        assert (benchmark)
        assert (isinstance(benchmark, PytorchCNN))
        assert (benchmark.name == 'pytorch-' + model)
        assert (benchmark.type == BenchmarkType.MODEL)

        # Check predefined parameters of resnet101 model.
        assert (benchmark._args.model_type == model)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.batch_size == 1)
        assert (benchmark._args.image_size == 224)
        assert (benchmark._args.num_classes == 5)
        assert (benchmark._args.num_warmup == 2)
        assert (benchmark._args.num_steps == 4)

        # Check Dataset.
        assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

        # Check results and metrics.
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        for metric in check_metrics:
            assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
            assert (len(benchmark.result[metric]) == benchmark.run_count)
