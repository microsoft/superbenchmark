# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the ComputationCommunicationOverlap benchmarks.

ComputationCommunicationOverlap benchmark is used to test the performance of single node
when communication and computation overlap and figure out the issued GPU.
If the node has performance downgrade on this context:
    -Currently, 2 computation kernels are supported: mul and matmul of the matrix.
    -Communication kernel: NCCL AllReduce.
    -Each GPU will run the computation kernel and communication kernels in pipeline.
        to achieve overlap in some degree, in every loop,
        they will fist launch a NCCL allreduce kernel async
        and then launch #ratio computation kernels at once.
"""

import os
import time

# TODO - add mechanism to import torch as needed according to docker
import torch

from superbench.common.utils import logger
from superbench.benchmarks import DistributedImpl, DistributedBackend, BenchmarkRegistry, ReturnCode
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
            help='The K dim of matmul (M, K) * (K, N) or The K dim of mul (M, K)*(M, K).',
        )
        self._parser.add_argument(
            '--m',
            type=int,
            default=640,
            required=False,
            help='The M dim of matmul (M, K) * (K, N)  or The m dim of mul (M, K)*(M, K).',
        )
        self._parser.add_argument(
            '--p',
            type=int,
            default=5000,
            required=False,
            help='The P dim of Tensor(P, Q) to transfer in NCCL AllReduce.',
        )
        self._parser.add_argument(
            '--q',
            type=int,
            default=1400,
            required=False,
            help='The Q dim of Tensor(P, Q) to transfer in NCCL AllReduce.',
        )
        self._parser.add_argument(
            '--ratio',
            type=int,
            default=30,
            required=False,
            help='The execution ratio number between computation kernel and NCCL kernel.',
        )
        self._parser.add_argument(
            '--kernel',
            type=ComputationKernelType,
            default=[ComputationKernelType.MUL, ComputationKernelType.MATMUL],
            nargs='+',
            required=False,
            help='Computation kernel type. E.g. {}.'.format(' '.join(ComputationKernelType.get_values())),
        )
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=64,
            required=False,
            help='The number of warmup step.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=2048,
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

        try:
            torch.distributed.init_process_group(backend=self._args.distributed_backend.value)
            self.__world_size = int(os.environ['WORLD_SIZE'])
            self.__local_rank = int(os.environ['LOCAL_RANK'])
            # if self.__world_size < 2:
            # raise Exception("Distributed env check error, WORLD_SIZE < 2")
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)
            torch.distributed.destroy_process_group()
            logger.error('Initialize distributed env failed - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        if torch.cuda.is_available():
            torch.cuda.set_device(self.__local_rank)

        return True

    def __kernel_nccl_pipeline(self, kernel, matA, matB, stages, message, times):
        """Computation and NCCL kernel pipeline with single GPU.

        Args:
            kernel (ComputationKernelType): the type of the computation kernel to run.
            matA (list[tensor]): the matrix list used in matmul or mul for every stage.
            matB (tensor): the matrix used in matmul.
            stages (int): the ratio number of computation kernel and communication kernel.
            message(tensor): the data used to be transferred through NCCL.
            times(int): number of times in one step to run.

        Return:
            True of False: if computation kernel type is invalid, return False, else, return True.
        """
        if kernel == ComputationKernelType.MUL:
            for i in range(times):
                torch.distributed.all_reduce(message, op=torch.distributed.ReduceOp.SUM, async_op=True)
                for stage in range(stages):
                    matA[stage].mul(matA[stage])
        elif kernel == ComputationKernelType.MATMUL:
            for i in range(times):
                torch.distributed.all_reduce(message, op=torch.distributed.ReduceOp.SUM, async_op=True)
                for stage in range(stages):
                    matA[stage].matmul(matB)
        else:
            logger.error('Unknown comoputation kernel type - benchmark: {}, type: {}.'.format(self._name, kernel))
            return False
        return True

    def _benchmark(self):
        """Implementation for benchmarking."""
        M = self._args.m
        K = self._args.k
        N = self._args.n
        P = self._args.p
        Q = self._args.q
        kernels = self._args.kernel
        if self.__local_rank == 0:
            logger.info(
                'Computation Communication Overlap - using {} GPUs,\
                matrix shape for computation: M={} K={} N={},\
                message tensor shape of NCCL = [{},{}],\
                ratio between computation kernel and NCCL kernel={}'.format(
                    self.__world_size, M, K, N, P, Q, self._args.ratio
                )
            )

        MatA = list()
        MatB = list()
        # Matrix A
        for _ in range(self._args.ratio):
            MatA.append(torch.randn(M, K).cuda())
        # Matrix B
        MatB = torch.randn(K, N).cuda()
        # message for NCCL to transport
        shape = [P, Q]
        message = torch.randn(*shape).cuda()

        for kernel in kernels:
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

            compute_metric = '{}_time'.format(kernel)
            compute_elapse_times = [(compute_end - start) * 1000 / self._args.num_steps]

            if not self._process_numeric_result(compute_metric, compute_elapse_times):
                return False

            logger.info(
                'Computation_communication_overlap - round: {0}, name: {1}, gpu: {2} kernel: {3}, cost: {4} ms'.format(
                    self._curr_run_index, self._name, self.__local_rank, kernel,
                    (compute_end - start) * 1000 / self._args.num_steps
                )
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
            torch.distributed.destroy_process_group()
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_DESTROY_FAILURE)
            logger.error('Post process failed - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        return True


BenchmarkRegistry.register_benchmark(
    'pytorch-computation-communication-overlap', ComputationCommunicationOverlap, parameters='--kernel mul matmul'
)
