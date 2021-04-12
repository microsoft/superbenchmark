# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Computation-Communication-Overlap benchmark."""

from functools import wraps
import torch
from tests.helper import decorator
import tests.benchmarks.utils as utils
from superbench.benchmarks import BenchmarkRegistry, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.computation_communication_overlap \
    import ComputationCommunicationOverlap, ComputationKernelType
from superbench.common.utils import logger


def skip_if_not_multigpu(func):
    """Multi-GPU tests requires at least 2 GPUS. Skip if this is not met."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return func(*args, **kwargs)
        message = 'Need at least {} CUDA devices'.format(2)
        logger.error('Device error -  message: {}.'.format(message))

    return wrapper


@decorator.cuda_test
@decorator.pytorch_test
@skip_if_not_multigpu
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

        assert (len(benchmark.raw_data) == benchmark.run_count * len(benchmark._args.kernel))
        assert (len(benchmark.result) == benchmark.run_count * len(benchmark._args.kernel))


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_computation_communication_overlap_fake_distributed():
    """Test pytorch-computation-communication-overlap benchmark on single gpu."""
    context = BenchmarkRegistry.create_benchmark_context(
        'computation-communication-overlap',
        parameters='--num_warmup 5 --num_steps 10 --ratio 5',
        framework=Framework.PYTORCH
    )
    utils.setup_simulated_ddp_distributed_env()
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

    assert (len(benchmark.raw_data) == benchmark.run_count * len(benchmark._args.kernel))
    assert (len(benchmark.result) == benchmark.run_count * len(benchmark._args.kernel))
    utils.clean_simulated_ddp_distributed_env()
