# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Computation-Communication-Overlap benchmark."""

import os
import torch
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Process
from superbench.benchmarks import BenchmarkRegistry, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.computation_communication_overlap \
    import ComputationCommunicationOverlap, ComputationKernelType
import socket
from contextlib import closing
from superbench.common.utils import logger
from functools import wraps


def get_free_port():
    """Get free port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup_simulated_ddp_distributed_env(world_size, local_rank, port):
    """Function to setup the simulated DDP distributed envionment variables."""
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)


def benchmark_in_one_process(context, world_size, local_rank, port, queue):
    """Function to setup env for DDP initialization and run the benchmark in each single process."""
    setup_simulated_ddp_distributed_env(world_size, local_rank, port)
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    # parser object must be removed becaues it can not be serialized.
    benchmark._parser = None
    queue.put(benchmark)
    clean_simulated_ddp_distributed_env()


def simulated_ddp_distributed_benchmark(context, world_size):
    """Function to run the benchmark on #world_size number of processes."""
    port = get_free_port()
    process_list = []
    multiprocessing.set_start_method('spawn')

    queue = multiprocessing.Queue()

    for rank in range(world_size):
        process = Process(target=benchmark_in_one_process, args=(context, world_size, rank, port, queue))
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()
    results = [queue.get(1) for p in process_list]
    return results


def setup_simulated_fake_distributed_env():
    """Function to setup the simulated non-distributed envionment variables."""
    setup_simulated_ddp_distributed_env(1, 0, get_free_port())


def clean_simulated_ddp_distributed_env():
    """Function to clean up the simulated DDP distributed envionment variables."""
    os.environ.pop('WORLD_SIZE')
    os.environ.pop('RANK')
    os.environ.pop('LOCAL_RANK')
    os.environ.pop('MASTER_ADDR')
    os.environ.pop('MASTER_PORT')


def skip_if_not_multigpu(func):
    """Multi-GPU tests requires at least 2 GPUS. Skip if this is not met."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return func(*args, **kwargs)
        message = 'Need at least {} CUDA devices'.format(2)
        logger.error('Device error -  message: {}.'.format(message))

    return wrapper


def skip_if_no_gpu(func):
    """Gpu tests require at least 1 GPUS. Skip if this is not met."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            return func(*args, **kwargs)

        message = 'No cuda device'
        logger.error('Device error -  message: {}.'.format(message))

    return wrapper


@skip_if_not_multigpu
def test_pytorch_computation_communication_overlap_normal():
    """Test pytorch-computation-communication-overlap benchmark on distributed normal case."""
    context = BenchmarkRegistry.create_benchmark_context(
        'computation-communication-overlap',
        parameters='--num_warmup 5 --num_steps 10 --ratio 5',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    setup_simulated_fake_distributed_env()
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


@skip_if_no_gpu
def test_pytorch_computation_communication_overlap_fake_distributed():
    """Test pytorch-computation-communication-overlap benchmark on single gpu."""
    world_size = 2
    context = BenchmarkRegistry.create_benchmark_context(
        'computation-communication-overlap',
        parameters='--num_warmup 5 --num_steps 10 --ratio 5',
        framework=Framework.PYTORCH
    )
    results = simulated_ddp_distributed_benchmark(context, world_size)
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
