# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for sharding-matmul benchmark."""

import tests.benchmarks.utils as utils
from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.sharding_matmul import ShardingMatmul, ShardingMode
from superbench.common.utils import network


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_sharding_matmul():
    """Test pytorch-sharding-matmul benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'sharding-matmul',
        platform=Platform.CUDA,
        parameters='--run_count 2 --num_steps 20',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    port = network.get_free_port()
    assert (port)
    utils.setup_simulated_ddp_distributed_env(1, 0, port)
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, ShardingMatmul))
    assert (benchmark.name == 'pytorch-sharding-matmul')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check predefined parameters of sharding-matmul benchmark.
    assert (benchmark._args.mode == [ShardingMode.ALLREDUCE, ShardingMode.ALLGATHER])

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.run_count == 2)
    assert (benchmark._args.num_steps == 20)

    # Check results and metrics.
    assert (benchmark.run_count == 2)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for metric in ['allreduce_time', 'allgather_time']:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)

    utils.clean_simulated_ddp_distributed_env()
