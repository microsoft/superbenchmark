# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Computation-Communication-Overlap benchmark."""

import unittest

from tests.helper import decorator
import tests.benchmarks.utils as utils
from superbench.benchmarks import BenchmarkRegistry, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.computation_communication_overlap \
    import ComputationCommunicationOverlap, ComputationKernelType
from superbench.common.utils import network


# TODO - replace unittest.skip("no multiple GPUs") to decorator of skipIfNoMultiGPUS
@unittest.skip('no multiple GPUs')
@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_computation_communication_overlap_normal():
    """Test pytorch-computation-communication-overlap benchmark on distributed normal case."""
    context = BenchmarkRegistry.create_benchmark_context(
        'computation-communication-overlap',
        parameters='--num_warmup 5 --num_steps 10 --ratio 5',
        framework=Framework.PYTORCH
    )
    world_size = 2
    assert (BenchmarkRegistry.is_benchmark_context_valid(context))
    results = utils.simulated_ddp_distributed_benchmark(context, world_size)
    assert (results)
    for benchmark in results:
        # Check basic information.
        assert (benchmark)
        assert (isinstance(benchmark, ComputationCommunicationOverlap))
        assert (benchmark.name == 'pytorch-computation-communication-overlap')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check predefined parameters of sharding-matmul benchmark.
        assert (benchmark._args.kernel == [ComputationKernelType.MUL, ComputationKernelType.MATMUL])

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.num_steps == 10)

        # Check results and metrics.
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        assert (len(benchmark.raw_data) == len(benchmark._args.kernel))
        assert (len(benchmark.result) == len(benchmark._args.kernel) + benchmark.default_metric_count)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_computation_communication_overlap_fake_distributed():
    """Test pytorch-computation-communication-overlap benchmark on single gpu."""
    context = BenchmarkRegistry.create_benchmark_context(
        'computation-communication-overlap',
        parameters='--num_warmup 5 --num_steps 10 --ratio 5',
        framework=Framework.PYTORCH
    )
    port = network.get_free_port()
    assert (port)
    utils.setup_simulated_ddp_distributed_env(1, 0, port)
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, ComputationCommunicationOverlap))
    assert (benchmark.name == 'pytorch-computation-communication-overlap')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check predefined parameters of sharding-matmul benchmark.
    assert (benchmark._args.kernel == [ComputationKernelType.MUL, ComputationKernelType.MATMUL])

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.num_steps == 10)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    assert (len(benchmark.raw_data) == len(benchmark._args.kernel))
    assert (len(benchmark.result) == len(benchmark._args.kernel) + benchmark.default_metric_count)
    utils.clean_simulated_ddp_distributed_env()
