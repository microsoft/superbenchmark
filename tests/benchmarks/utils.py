# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for benchmark tests."""

import os
import socket
from contextlib import closing
import multiprocessing as multiprocessing
from multiprocessing import Process
from superbench.benchmarks import BenchmarkRegistry


def setup_simulated_ddp_distributed_env():
    """Function to setup the simulated DDP distributed envionment variables."""
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'


def clean_simulated_ddp_distributed_env():
    """Function to clean up the simulated DDP distributed envionment variables."""
    os.environ.pop('WORLD_SIZE')
    os.environ.pop('RANK')
    os.environ.pop('LOCAL_RANK')
    os.environ.pop('MASTER_ADDR')
    os.environ.pop('MASTER_PORT')


def get_free_port():
    """Get free port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup_simulated_ddp_distributed_env_custom(world_size, local_rank, port):
    """Function to setup the simulated DDP distributed envionment variables."""
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)


def benchmark_in_one_process(context, world_size, local_rank, port, queue):
    """Function to setup env for DDP initialization and run the benchmark in each single process."""
    setup_simulated_ddp_distributed_env_custom(world_size, local_rank, port)
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
