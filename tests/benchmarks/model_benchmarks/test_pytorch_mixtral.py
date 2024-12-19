# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mixtral model benchmarks."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_mixtral import PytorchMixtral


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_mixtral_8x7b():
    """Test pytorch-mixtral-8x7b benchmark for fp16 train and inference."""
    context = BenchmarkRegistry.create_benchmark_context(
        'mixtral-8x7b',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision fp8_e4m3 \
            --model_action inference',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchMixtral))
    assert (benchmark.name == 'pytorch-mixtral-8x7b')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of mixtral2 7b model.
    assert (benchmark._args.hidden_size == 4096)
    assert (benchmark._args.num_hidden_layers == 32)
    assert (benchmark._args.num_attention_heads == 32)
    assert (benchmark._args.num_key_value_heads == 8)
    assert (benchmark._args.intermediate_size == 14336)
    assert (benchmark._args.max_position_embeddings == 32768)
    assert (benchmark._args.router_aux_loss_coef == 0.02)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 100)
    assert (benchmark._args.seq_len == 32)
    assert (benchmark._args.num_warmup == 1)
    assert (benchmark._args.num_steps == 2)

    # Test Dataset.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    for metric in [ 'fp8_e4m3_inference_step_time', 'fp8_e4m3_inference_throughput']:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)
