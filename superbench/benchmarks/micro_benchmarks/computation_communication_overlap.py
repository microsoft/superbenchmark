# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the ComputationCommunicationOverlap benchmarks.

ComputationCommunicationOverlap benchmark is used to test the performance of single node when communication and computation overlap and figure out the issued GPU if the node has performance downgrade on this context:
    -Currently, 2 computation kernels are supported: mul and matmul of the matrix
    -Communication kernel: NCCL AllReduce
    -Each GPU will run the computation kernel and communication kernels in pipeline to achieve overlap in some degree, in every loop, they will fist launch a nccl allreduce kernel async and then launch #ratio computation kernels at once.
"""

import os
import time

# TODO - add mechanism to import torch as needed according to docker
import torch

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmark
from superbench.benchmarks.context import Enum


class ComputationKernelType(Enum):
    """The Enum class representing different computation kernel type."""
    MATMUL = 'matmul'
    MUL = 'mul'


class ComputationCommunicationOverlap(MicroBenchmark):
    """The base class of micro-benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        # Command lines to launch the micro-benchmarks.
        self.__commands = list()
        self.__world_size = 1
        self.__local_rank = 0
        torch.backends.cudnn.benchmark = True

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--n',
            type=int,
            default=1880,
            required=False,
            help='The N dim of matmul (M, K) * (K, N).',
        )
        self._parser.add_argument(
            '--k',
            type=int,
            default=1024,
            required=False,
            help='The K dim of matmul (M, K) * (K, N) or The K dim or mul (M, K)*(M, K). ',
        )
        self._parser.add_argument(
            '--m',
            type=int,
            default=640,
            required=False,
            help='The M dim of matmul (M, K) * (K, N)  or The m dim or mul (M, K)*(M, K). ',
        )
        self._parser.add_argument(
            '--p',
            type=int,
            default=5000,
            required=False,
            help='The P dim of Tensor(P, Q) to transfer in Nccl AllReduce.',
        )
        self._parser.add_argument(
            '--q',
            type=int,
            default=1400,
            required=False,
            help='The Q dim of Tensor(P, Q) to transfer in Nccl AllReduce.',
        )
        self._parser.add_argument(
            "--ratio",
            type=int,
            default=30,
            required=False,
            help="The execution ratio number between computation kernel and nccl kernel",
        )
        self._parser.add_argument(
            '--kernel',
            type=ComputationKernelType,
            default=[ComputationKernelType.MUL],
            required=False,
            help='Computation kernel type. E.g. {}.'.format(' '.join(ComputationKernelType.get_values())),
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
            default=2000,
            required=False,
            help='The number of test step.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        try:
            torch.distributed.init_process_group(backend='nccl')
            self.__world_size = int(os.environ['WORLD_SIZE'])
            self.__local_rank = int(os.environ['LOCAL_RANK'])
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)
            logger.error('Initialize distributed env failed - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        if torch.cuda.is_available():
            torch.cuda.set_device(self.__local_rank)

        return True

    def __kernel_nccl_pipeline(self, kernel, matA, matB, stages, message, times):
        """computation and nccl kernel pipeline with single GPU.

        Args:
            kernel (ComputationKernelType): the type of the computation kernel to run
            matA (list[tensor]): the matrix list used in matmul or mul for every stage
            matB (tensor): the matrix used in matmul
            stages (int): the ratio number of computation kernel and communication kernel
            message(tensor): the data used to be transfered through NCCL
            times(int): number of times in one step to run

        Return:
            True of False: if computation kernel type is invalid, return False, else, return True
        """

        for i in range(times):
            if self.__world_size > 1:
                torch.distributed.all_reduce(message, op=torch.distributed.ReduceOp.SUM, async_op=True)
            for stage in range(stages):
                if kernel == ComputationKernelType.MUL:
                    matC = matA[stage].mul(matA[stage])
                elif kernel == ComputationKernelType.MATMUL:
                    matC = matA[stage].matmul(matB)
                else:
                    logger.error(
                        'Unknown comoputation kernel type - benchmark: {}, type: {}.'.format(self._name, kernel)
                    )
                    return False
        return True

    def _benchmark(self):
        """Implementation for benchmarking."""
        M = self._args.m
        K = self._args.k
        N = self._args.n
        P = self._args.p
        Q = self._args.q
        kernel = self._args.kernel
        if self.__local_rank == 0:
            logger.info(
                "Computation Communication Overlap - using {} GPUs, matrix shape for computation: M={} K={} N={}, message tensor shape of nccl = [{},{}], ratio between computation kernel and nccl kernel={}"
                .format(self.__world_size, M, K, N, P, Q, self._args.ratio)
            )

        MatA = list()
        MatB = list()
        # Matrix A
        for _ in range(self._args.ratio):
            MatA.append(torch.randn(M, K).cuda())
        # Matrix B
        MatB = torch.randn(K, N).cuda()
        # message for nccl to transport
        shape = [P, Q]
        message = torch.randn(*shape).cuda()

        # warm up
        for i in range(self._args.num_warmup):
            if not self.__kernel_nccl_pipeline(kernel, MatA, MatB, self._args.ratio, message, times=100):
                return False
        torch.cuda.synchronize()

        # run and collect results
        start = time.perf_counter()
        for i in range(self._args.num_steps):
            self.__kernel_nccl_pipeline(kernel, MatA, MatB, self._args.ratio, message, times=100)
        compute_end = time.perf_counter()
        torch.cuda.synchronize()
        end = time.perf_counter()

        logger.info(
            "GPU {} total compute cost {}".format(
                self.__local_rank, (compute_end - start) * 1000 / self._args.num_steps
            )
        )

        compute_metric = 'computation_communication_overlap_gpu{}_computation_cost_{}'.format(self.__local_rank, kernel)
        total_metric = 'computation_communication_overlap_gpu{}_total_cost_{}'.format(self.__local_rank, kernel)
        compute_elapse_times = [(compute_end - start) * 1000 / self._args.num_steps]
        elapse_times = [(end - start) * 1000 / self._args.num_steps]
        if not self._process_numeric_result(compute_metric, compute_elapse_times
                                            ) or not self._process_numeric_result(total_metric, elapse_times):
            return False

        logger.info(
            " GPU: {} {} pipeline mean cost of each stage with NCCL all_reduce, time: {} ms".format(
                self.__local_rank, kernel, (end - start) * 1000 / self._args.num_steps
            )
        )
        torch.distributed.destroy_process_group()
        return True


BenchmarkRegistry.register_benchmark(
    'pytorch-computation-communication-overlap-matmul', ComputationCommunicationOverlap, parameters='--kernel matmul'
)
BenchmarkRegistry.register_benchmark(
    'pytorch-computation-communication-overlap-mul', ComputationCommunicationOverlap, parameters='--kernel mul'
)
