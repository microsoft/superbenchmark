# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GPT2 model benchmarks."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import PytorchGPT2


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_gpt2_small():
    """Test pytorch-gpt2-small benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'gpt2-small',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 4 --num_steps 16',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchGPT2))
    assert (benchmark.name == 'pytorch-gpt2-small')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of gpt2-large model.
    assert (benchmark._args.hidden_size == 768)
    assert (benchmark._args.num_hidden_layers == 12)
    assert (benchmark._args.num_attention_heads == 12)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 5)
    assert (benchmark._args.seq_len == 8)
    assert (benchmark._args.num_warmup == 4)
    assert (benchmark._args.num_steps == 16)

    # Test Dataset.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for metric in [
        'steptime_train_float32', 'throughput_train_float32', 'steptime_train_float16', 'throughput_train_float16'
    ]:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)
