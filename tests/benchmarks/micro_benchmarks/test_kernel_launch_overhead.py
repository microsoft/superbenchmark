# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for kernel-launch benchmark."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode


@decorator.cuda_test
def test_pytorch_matmul():
    """Test kernel-launch benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'kernel-launch', parameters='--num_warmup 200 --num_steps 20000 --interval 100'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'kernel-launch')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.num_warmup == 200)
    assert (benchmark._args.num_steps == 20000)
    assert (benchmark._args.interval == 100)

    # Check results and metrics.
    # TODO - will change the result checking after kernel launch source part is merged.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_BINARY_NOT_EXIST)
