# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for distributed inference simulation benchmark."""

import unittest

from tests.helper import decorator
import tests.benchmarks.utils as utils
from superbench.benchmarks import BenchmarkRegistry, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.dist_inference_simulation \
    import DistInferenceSimulation, ComputationKernelType, CommunicationKernelType, ActivationKernelType
from superbench.common.utils import network


# TODO - replace unittest.skip("no multiple GPUs") to decorator of skipIfNoMultiGPUS
@unittest.skip('no multiple GPUs')
@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_dist_inference_simulation_normal():
    """Test pytorch-dist-inference-simulation benchmark on distributed normal case."""
    context = BenchmarkRegistry.create_benchmark_context(
        'pytorch-dist-inference-simulation',
        parameters='',
        framework=Framework.PYTORCH
    )
    world_size = 2
    assert (BenchmarkRegistry.is_benchmark_context_valid(context))
    results = utils.simulated_ddp_distributed_benchmark(context, world_size)
    assert (results)
    for benchmark in results:
        # Check basic information.
        assert (benchmark)
        assert (isinstance(benchmark, DistInferenceSimulation))
        assert (benchmark.name == 'pytorch-dist-inference-simulation')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check predefined parameters of dist-inference-simulation benchmark.
        assert (benchmark._args.batch_size == benchmark.__default_batch_size)
        assert (benchmark._args.input_size == benchmark.__default_input_size)
        assert (benchmark._args.hidden_size == benchmark.__default_hidden_size)
        assert (benchmark._args.num_layers == benchmark.__default_num_layers)
        assert (benchmark._args.computation_kernel == benchmark.__default_computation_kernel)
        assert (benchmark._args.communication_kernel == benchmark.__default_communication_kernel)
        assert (benchmark._args.activation_kernel == benchmark.__default_activation_kernel)
        assert (benchmark._args.precision == benchmark.__default_precision)
        assert (benchmark._args.num_warmup == benchmark.__default_num_warmup)
        assert (benchmark._args.num_steps == benchmark.__default_num_steps)
        assert (benchmark._args.distributed_impl == benchmark.__default_distributed_impl)
        assert (benchmark._args.distributed_backend == benchmark.__default_distributed_backend)

        # Check results and metrics.
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (len(benchmark.raw_data) == 1)
        # warmup_step_times and test_step_times
        assert (len(benchmark.result) == 2)

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_dist_inference_simulation_fake_distributed():
    """Test pytorch-dist-inference-simulation benchmark on single gpu."""
    context = BenchmarkRegistry.create_benchmark_context(
        'pytorch-dist-inference-simulation',
        parameters='',
        framework=Framework.PYTORCH
    )
    port = network.get_free_port()
    assert (port)
    utils.setup_simulated_ddp_distributed_env(1, 0, port)
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, DistInferenceSimulation))
    assert (benchmark.name == 'pytorch-dist-inference-simulation')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check predefined parameters of dist-inference-simulation benchmark.
    assert (benchmark._args.batch_size == benchmark.__default_batch_size)
    assert (benchmark._args.input_size == benchmark.__default_input_size)
    assert (benchmark._args.hidden_size == benchmark.__default_hidden_size)
    assert (benchmark._args.num_layers == benchmark.__default_num_layers)
    assert (benchmark._args.computation_kernel == benchmark.__default_computation_kernel)
    assert (benchmark._args.communication_kernel == benchmark.__default_communication_kernel)
    assert (benchmark._args.activation_kernel == benchmark.__default_activation_kernel)
    assert (benchmark._args.precision == benchmark.__default_precision)
    assert (benchmark._args.num_warmup == benchmark.__default_num_warmup)
    assert (benchmark._args.num_steps == benchmark.__default_num_steps)
    assert (benchmark._args.distributed_impl == benchmark.__default_distributed_impl)
    assert (benchmark._args.distributed_backend == benchmark.__default_distributed_backend)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (len(benchmark.raw_data) == 1)
    # warmup_step_times and test_step_times
    assert (len(benchmark.result) == 2)

    utils.clean_simulated_ddp_distributed_env()
