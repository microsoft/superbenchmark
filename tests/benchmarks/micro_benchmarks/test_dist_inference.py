# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for distributed inference benchmark."""

import unittest

from tests.helper import decorator
import tests.benchmarks.utils as utils
from superbench.benchmarks \
    import BenchmarkRegistry, Framework, BenchmarkType, ReturnCode, Precision, DistributedImpl, DistributedBackend
from superbench.benchmarks.micro_benchmarks.dist_inference \
    import DistInference, ComputationKernelType, CommunicationKernelType, ActivationKernelType
from superbench.common.utils import network


# TODO - replace unittest.skip("no multiple GPUs") to decorator of skipIfNoMultiGPUS
@unittest.skip('no multiple GPUs')
@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_dist_inference_normal():
    """Test pytorch-dist-inference benchmark on distributed normal case."""
    context = BenchmarkRegistry.create_benchmark_context('dist-inference', parameters='', framework=Framework.PYTORCH)
    world_size = 2
    assert (BenchmarkRegistry.is_benchmark_context_valid(context))
    results = utils.simulated_ddp_distributed_benchmark(context, world_size)
    assert (results)
    for benchmark in results:
        # Check basic information.
        assert (benchmark)
        assert (isinstance(benchmark, DistInference))
        assert (benchmark.name == 'pytorch-dist-inference')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check predefined parameters of dist-inference benchmark.
        assert (benchmark._args.use_pytorch is True)
        assert (benchmark._args.batch_size == 64)
        assert (benchmark._args.input_size == 1024)
        assert (benchmark._args.hidden_size == 1024)
        assert (benchmark._args.alpha == 1.0)
        assert (benchmark._args.beta == 1.0)
        assert (benchmark._args.num_layers == 1)
        assert (benchmark._args.computation_kernel == ComputationKernelType.MATMUL)
        assert (benchmark._args.communication_kernel == CommunicationKernelType.ALLREDUCE)
        assert (benchmark._args.activation_kernel == ActivationKernelType.RELU)
        assert (benchmark._args.precision == Precision.FLOAT32)
        assert (benchmark._args.num_warmup == 50)
        assert (benchmark._args.num_steps == 10000)
        assert (benchmark._args.distributed_impl == DistributedImpl.DDP)
        assert (benchmark._args.distributed_backend == DistributedBackend.NCCL)
        assert (benchmark._args.use_cuda_graph is False)

        # Check results and metrics.
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        # step_times
        assert (len(benchmark.raw_data) == 1)
        # return code + (avg, 50th, 90th, 95th, 99th, 99.9th)
        assert (len(benchmark.result) == 7)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_dist_inference_fake_distributed():
    """Test pytorch-dist-inference benchmark on single gpu."""
    context = BenchmarkRegistry.create_benchmark_context('dist-inference', parameters='', framework=Framework.PYTORCH)
    port = network.get_free_port()
    assert (port)
    utils.setup_simulated_ddp_distributed_env(1, 0, port)
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, DistInference))
    assert (benchmark.name == 'pytorch-dist-inference')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check predefined parameters of dist-inference benchmark.
    assert (benchmark._args.use_pytorch is True)
    assert (benchmark._args.batch_size == 64)
    assert (benchmark._args.input_size == 1024)
    assert (benchmark._args.hidden_size == 1024)
    assert (benchmark._args.alpha == 1.0)
    assert (benchmark._args.beta == 1.0)
    assert (benchmark._args.num_layers == 1)
    assert (benchmark._args.computation_kernel == ComputationKernelType.MATMUL)
    assert (benchmark._args.communication_kernel == CommunicationKernelType.ALLREDUCE)
    assert (benchmark._args.activation_kernel == ActivationKernelType.RELU)
    assert (benchmark._args.precision == Precision.FLOAT32)
    assert (benchmark._args.num_warmup == 50)
    assert (benchmark._args.num_steps == 10000)
    assert (benchmark._args.distributed_impl == DistributedImpl.DDP)
    assert (benchmark._args.distributed_backend == DistributedBackend.NCCL)
    assert (benchmark._args.use_cuda_graph is False)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    # step_times
    assert (len(benchmark.raw_data) == 1)
    # return code + (avg, 50th, 90th, 95th, 99th, 99.9th)
    assert (len(benchmark.result) == 7)

    utils.clean_simulated_ddp_distributed_env()
