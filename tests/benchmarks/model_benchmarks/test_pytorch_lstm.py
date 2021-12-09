# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for LSTM model benchmarks."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_lstm import PytorchLSTM


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_lstm_with_gpu():
    """Test pytorch-lstm benchmark with GPU."""
    run_pytorch_lstm(
        parameters='--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
            --model_action train inference',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp16_train_step_time', 'fp16_train_throughput',
            'fp32_inference_step_time', 'fp32_inference_throughput', 'fp16_inference_step_time',
            'fp16_inference_throughput'
        ]
    )


@decorator.pytorch_test
def test_pytorch_lstm_no_gpu():
    """Test pytorch-lstm benchmark with CPU."""
    run_pytorch_lstm(
        parameters='--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
            --model_action train inference --precision float32 --no_gpu',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp32_inference_step_time', 'fp32_inference_throughput'
        ]
    )


def run_pytorch_lstm(parameters='', check_metrics=[]):
    """Test pytorch-lstm benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'lstm', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchLSTM))
    assert (benchmark.name == 'pytorch-lstm')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of lstm model.
    assert (benchmark._args.input_size == 256)
    assert (benchmark._args.hidden_size == 1024)
    assert (benchmark._args.num_layers == 8)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 5)
    assert (benchmark._args.seq_len == 8)
    assert (benchmark._args.num_warmup == 2)
    assert (benchmark._args.num_steps == 4)

    # Check dataset scale.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for metric in check_metrics:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)
