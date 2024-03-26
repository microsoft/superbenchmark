# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for distributed inference benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
import tests.benchmarks.utils as utils
from superbench.benchmarks \
    import BenchmarkRegistry, Framework, BenchmarkType, ReturnCode, Precision, DistributedImpl, DistributedBackend, \
    Platform
from superbench.benchmarks.micro_benchmarks.dist_inference \
    import DistInference, ComputationKernelType, CommunicationKernelType, ActivationKernelType
from superbench.common.utils import network


# TODO - replace unittest.skip("no multiple GPUs") to decorator of skipIfNoMultiGPUS
@unittest.skip('no multiple GPUs')
@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_dist_inference_normal():
    """Test pytorch-dist-inference benchmark on distributed normal case."""
    context = BenchmarkRegistry.create_benchmark_context(
        'dist-inference', parameters='--use_pytorch', framework=Framework.PYTORCH
    )
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
        assert (benchmark._args.tune_gemm is False)

        # Check results and metrics.
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        # step_times
        assert (len(benchmark.raw_data) == 1)
        # return code + (avg, 50th, 90th, 95th, 99th, 99.9th)
        assert (7 == len(benchmark.result))


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_dist_inference_fake_distributed():
    """Test pytorch-dist-inference benchmark on single gpu."""
    context = BenchmarkRegistry.create_benchmark_context(
        'dist-inference', parameters='--use_pytorch', framework=Framework.PYTORCH
    )
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
    assert (benchmark._args.tune_gemm is False)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    # step_times
    assert (len(benchmark.raw_data) == 1)
    # return code + (avg, 50th, 90th, 95th, 99th, 99.9th)
    assert (len(benchmark.result) == 7)

    utils.clean_simulated_ddp_distributed_env()


class DistInferenceCppImplTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for pytorch-dist-inference benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/dist_inference'])

    def _test_dist_inference_command_generation(self, platform):
        """Test pytorch-dist-inference cpp impl benchmark command generation."""
        benchmark_name = 'pytorch-dist-inference'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)

        batch_size = 1
        input_size = 2
        hidden_size = 3
        alpha = 4.0
        beta = 5.0
        num_layers = 6
        num_warmup = 7
        num_steps = 8
        wrapper_params_format_str = \
            '--batch_size %d --input_size %d --hidden_size %d ' \
            '--alpha %g --beta %g --num_layers %d --num_warmup %d --num_steps %d --use_cuda_graph --tune_gemm'
        parameters = wrapper_params_format_str % (
            batch_size, input_size, hidden_size, alpha, beta, num_layers, num_warmup, num_steps
        )
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.use_pytorch is False)
        assert (benchmark._args.batch_size == batch_size)
        assert (benchmark._args.input_size == input_size)
        assert (benchmark._args.hidden_size == hidden_size)
        assert (benchmark._args.alpha == alpha)
        assert (benchmark._args.beta == beta)
        assert (benchmark._args.num_layers == num_layers)
        assert (benchmark._args.num_warmup == num_warmup)
        assert (benchmark._args.num_steps == num_steps)
        assert (benchmark._args.use_cuda_graph is True)
        assert (benchmark._args.tune_gemm is True)

        # Check command
        assert (1 == len(benchmark._commands))
        for cmd in benchmark._commands:
            m, n, k = hidden_size, batch_size, input_size
            bench_params_format_str = \
                '%s -m %d -n %d -k %d --alpha %g --beta %g ' + \
                '--num_layers %d --num_warmups %d --num_iters %d --use_cuda_graph --tune_gemm'
            assert (
                cmd == (
                    bench_params_format_str %
                    (benchmark._DistInference__bin_path, m, n, k, alpha, beta, num_layers, num_warmup, num_steps)
                )
            )

    @decorator.cuda_test
    def test_dist_inference_command_generation_cuda(self):
        """Test pytorch-dist-inference cpp impl benchmark command generation, CUDA case."""
        self._test_dist_inference_command_generation(Platform.CUDA)

    @decorator.rocm_test
    def test_dist_inference_command_generation_rocm(self):
        """Test pytorch-dist-inference cpp impl benchmark command generation, ROCm case."""
        self._test_dist_inference_command_generation(Platform.ROCM)

    @decorator.load_data('tests/data/dist_inference.log')
    def _test_dist_inference_result_parsing(self, platform, test_raw_output):
        """Test pytorch-dist-inference cpp impl benchmark result parsing."""
        benchmark_name = 'pytorch-dist-inference'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'pytorch-dist-inference')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Positive case - valid raw output.
        assert (benchmark._process_raw_result(0, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # step_times
        assert (len(benchmark.raw_data) == 2)
        # return code + (avg, 50th, 90th, 95th, 99th, 99.9th)
        assert (7 == len(benchmark.result))
        assert (benchmark.result['return_code'] == [0])
        assert (benchmark.result['step_times'] == [1.9052048])
        assert (benchmark.result['step_times_50'] == [1.851])
        assert (benchmark.result['step_times_90'] == [1.89637])
        assert (benchmark.result['step_times_95'] == [2.12037])
        assert (benchmark.result['step_times_99'] == [2.67155])
        assert (benchmark.result['step_times_99.9'] == [4.4198])

        # Negative case - invalid raw output.
        assert (benchmark._process_raw_result(1, 'Latency of step: xxx ms') is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)

    @decorator.cuda_test
    def test_dist_inference_result_parsing_cuda(self):
        """Test pytorch-dist-inference cpp impl benchmark result parsing, CUDA case."""
        self._test_dist_inference_result_parsing(Platform.CUDA)

    @decorator.rocm_test
    def test_dist_inference_result_parsing_rocm(self):
        """Test pytorch-dist-inference cpp impl benchmark result parsing, ROCm case."""
        self._test_dist_inference_result_parsing(Platform.ROCM)
