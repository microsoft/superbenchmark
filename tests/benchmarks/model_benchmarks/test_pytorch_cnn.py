# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for CNN model benchmarks."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_cnn import PytorchCNN


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_cnn():
    """Test pytorch cnn benchmarks."""
    for model in [
        'resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19'
    ]:
        context = BenchmarkRegistry.create_benchmark_context(
            model,
            platform=Platform.CUDA,
            parameters='--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 \
                --model_action train inference',
            framework=Framework.PYTORCH
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
        for metric in [
            'steptime_train_float32', 'throughput_train_float32', 'steptime_train_float16', 'throughput_train_float16',
            'steptime_inference_float32', 'throughput_inference_float32', 'steptime_inference_float16',
            'throughput_inference_float16'
        ]:
            assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
            assert (len(benchmark.result[metric]) == benchmark.run_count)
