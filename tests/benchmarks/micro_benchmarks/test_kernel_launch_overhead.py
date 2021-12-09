# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for kernel-launch benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode


@decorator.cuda_test
def test_kernel_launch_overhead():
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
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output_0' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_0']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_0'][0], str))
    for metric in ['event_time', 'wall_time']:
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
