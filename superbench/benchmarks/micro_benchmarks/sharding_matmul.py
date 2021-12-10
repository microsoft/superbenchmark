# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the ShardingMatmul benchmarks.

ShardingMatmul benchmark is used to test the performance of large scale matmul operation with multiple GPUs:
  allreduce: Each GPU will calculate part of the MM calculation, and use AllReduce to merge all data into one tensor.
  allgather: Each GPU will calculate part of the MM calculation, and use AllGather + Concat to merge all data into
   one tensor.
  nosharding: Pure matmul operation with one GPU.

"""

import os
import time
import statistics

# TODO - add mechanism to import torch as needed according to docker.
import torch

from superbench.common.utils import logger
from superbench.benchmarks import DistributedImpl, DistributedBackend, BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmark
from superbench.benchmarks.context import Enum
from superbench.benchmarks.reducer import ReduceType


class ShardingMode(Enum):
    """The Enum class representing different sharding mode."""
    ALLREDUCE = 'allreduce'
    ALLGATHER = 'allgather'
    NOSHARDING = 'nosharding'


class ShardingMatmul(MicroBenchmark):
    """The base class of micro-benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self.__world_size = 1
        self.__local_rank = 0
        torch.backends.cudnn.benchmark = True

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--n',
            type=int,
            default=12288,
            required=False,
            help='The N dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--k',
            type=int,
            default=12288,
            required=False,
            help='The K dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--m',
            type=int,
            default=16000,
            required=False,
            help='The M dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--mode',
            type=ShardingMode,
            default=[ShardingMode.NOSHARDING],
            nargs='+',
            required=False,
            help='Sharding modes. E.g. {}.'.format(' '.join(ShardingMode.get_values())),
        )
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=10,
            required=False,
            help='The number of warmup step.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=500,
            required=False,
            help='The number of test step.',
        )
        self._parser.add_argument(
            '--distributed_impl',
            type=DistributedImpl,
            default=DistributedImpl.DDP,
            required=False,
            help='Distributed implementations. E.g. {}.'.format(' '.join(DistributedImpl.get_values())),
        )

        self._parser.add_argument(
            '--distributed_backend',
            type=DistributedBackend,
            default=DistributedBackend.NCCL,
            required=False,
            help='Distributed backends. E.g. {}.'.format(' '.join(DistributedBackend.get_values())),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if self._args.distributed_impl != DistributedImpl.DDP:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)
            logger.error(
                'Unsupported distributed implementation - model: {}, distributed implementation: {}.'.format(
                    self._name, self._args.distributed_impl
                )
            )
            return False

        if ShardingMode.ALLGATHER in self._args.mode or ShardingMode.ALLREDUCE in self._args.mode:
            try:
                torch.distributed.init_process_group(backend='nccl')
                self.__world_size = int(os.environ['WORLD_SIZE'])
                self.__local_rank = int(os.environ['LOCAL_RANK'])
            except BaseException as e:
                self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)
                torch.distributed.destroy_process_group()
                logger.error(
                    'Initialize distributed env failed - benchmark: {}, message: {}.'.format(self._name, str(e))
                )
                return False

        if torch.cuda.is_available():
            torch.cuda.set_device(self.__local_rank)

        return True

    def __matmul_nosharding(self, M, K, N):
        """Matmul with single GPU.

        Args:
            N (int): The N dim of matmul (N, K) * (K, M).
            K (int): The K dim of matmul (N, K) * (K, M).
            M (int): The M dim of matmul (N, K) * (K, M).

        Return:
            elapse_times (List[float]): cost of every test.
        """
        x = torch.ones(N, K).cuda()
        y = torch.ones(K, M).cuda()
        for i in range(self._args.num_warmup):
            torch.matmul(x, y)
            torch.cuda.synchronize()

        elapse_times = list()
        for i in range(self._args.num_steps):
            start = time.time()
            torch.matmul(x, y)
            torch.cuda.synchronize()
            end = time.time()
            elapse_times.append((end - start) * 1000)

        return elapse_times

    def __matmul_allreduce(self, M, K, N):
        """Matmul with allreduce sharding.

        Args:
            N (int): The N dim of matmul (N, K) * (K, M).
            K (int): The K dim of matmul (N, K) * (K, M).
            M (int): The M dim of matmul (N, K) * (K, M).

        Return:
            elapse_times (List[float]): cost of every test.
        """
        x = torch.ones(N, K // self.__world_size).cuda()
        y = torch.ones(K // self.__world_size, M).cuda()
        for i in range(self._args.num_warmup):
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.SUM)
            torch.cuda.synchronize()

        elapse_times = list()
        for i in range(self._args.num_steps):
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            torch.distributed.all_reduce(z, op=torch.distributed.ReduceOp.SUM)
            torch.cuda.synchronize()
            end = time.time()
            elapse_times.append((end - start) * 1000)

        return elapse_times

    def __matmul_allgather(self, M, K, N):
        """Matmul with allgather sharding.

        Args:
            N (int): The N dim of matmul (N, K) * (K, M).
            K (int): The K dim of matmul (N, K) * (K, M).
            M (int): The M dim of matmul (N, K) * (K, M).

        Return:
            elapse_times (List[float]): cost of every test.
        """
        x = torch.ones(N // self.__world_size, K).cuda()
        y = torch.ones(K, M).cuda()

        tensor_list = list()
        for i in range(self.__world_size):
            tensor_list.append(torch.zeros(N // self.__world_size, M).cuda())

        for i in range(self._args.num_warmup):
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            torch.distributed.all_gather(tensor_list, z)
            torch.cuda.synchronize()

        elapse_times = list()
        for i in range(self._args.num_steps):
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            torch.distributed.all_gather(tensor_list, z)
            z = torch.cat(tensor_list, 0)
            torch.cuda.synchronize()
            end = time.time()
            elapse_times.append((end - start) * 1000)

        return elapse_times

    def _benchmark(self):
        """Implementation for benchmarking."""
        M = self._args.m
        K = self._args.k
        N = self._args.n
        for mode in self._args.mode:
            if mode == ShardingMode.NOSHARDING:
                elapse_times = self.__matmul_nosharding(M, K, N)
            elif mode == ShardingMode.ALLREDUCE:
                elapse_times = self.__matmul_allreduce(M, K, N)
            elif mode == ShardingMode.ALLGATHER:
                elapse_times = self.__matmul_allgather(M, K, N)
            else:
                logger.error('Unknown sharding mode - benchmark: {}, mode: {}.'.format(self._name, mode))
                return False

            metric = '{}_time'.format(mode)
            if not self._process_numeric_result(metric, elapse_times, reduce_type=ReduceType.MAX):
                return False

            logger.info(
                'Matmul sharding - round: {0}, name: {1}, shape: ({2}, {3}) * ({3}, {4}), mode: {5}, cost: {6} ms'.
                format(self._curr_run_index, self._name, M, K, N, mode, statistics.mean(elapse_times))
            )

        return True

    def _postprocess(self):
        """Postprocess/cleanup operations after the benchmarking.

        Return:
            True if _postprocess() succeed.
        """
        if not super()._postprocess():
            return False

        try:
            if ShardingMode.ALLGATHER in self._args.mode or ShardingMode.ALLREDUCE in self._args.mode:
                torch.distributed.destroy_process_group()
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_DESTROY_FAILURE)
            logger.error(
                'Post process failed - benchmark: {}, mode: {}, message: {}.'.format(
                    self._name, self._args.mode, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('pytorch-sharding-matmul', ShardingMatmul, parameters='--mode allreduce allgather')
BenchmarkRegistry.register_benchmark('pytorch-matmul', ShardingMatmul, parameters='--mode nosharding')
