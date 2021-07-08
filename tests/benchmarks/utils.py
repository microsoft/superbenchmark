# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for benchmark tests."""

import os
import multiprocessing as multiprocessing
from multiprocessing import Process

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import network


def clean_simulated_ddp_distributed_env():
    """Function to clean up the simulated DDP distributed envionment variables."""
    os.environ.pop('WORLD_SIZE')
    os.environ.pop('RANK')
    os.environ.pop('LOCAL_RANK')
    os.environ.pop('MASTER_ADDR')
    os.environ.pop('MASTER_PORT')


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
    """Function to run the benchmark on #world_size number of processes.

    Return:
        results (list): list of benchmark results from #world_size number of processes.
    """
    port = network.get_free_port()
    if not port:
        return None
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
