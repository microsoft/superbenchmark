# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the distributed inference simulation model."""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from superbench.common.utils import logger
from superbench.benchmarks import DistributedImpl, DistributedBackend, BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmark
from superbench.benchmarks.context import Enum

class ComputationKernelType(Enum):
    """The Enum class representing different computation kernel type."""
    ADDMM = 'addmm'
    LINEAR = 'linear'
    MATMUL = 'matmul'
    MUL = 'mul'

class CommunicationKernelType(Enum):
    """The Enum class representing different communication kernel type."""
    ALLGATHER = 'allgather'
    ALLREDUCE = 'allreduce'
    ALLTOALL = 'alltoall'

class ActivationKernelType(Enum):
    """The Enum class representing different activation kernel type."""
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'

class DistInferenceSimulationModel():
    def __init__(self, input_size, hidden_size, num_layers, precision, computation, communication, activation, device, num_ranks):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.linear = nn.Linear(self.input_size, self.hidden_size, device=self.device)
        self.weights = torch.rand(self.input_size, self.hidden_size, dtype=getattr(torch, precision.value), device=self.device)
        self.bias = torch.rand(self.hidden_size, dtype=getattr(torch, precision.value), device=self.device)

        self.num_ranks = num_ranks

        self.computation_kernel = None
        if computation == ComputationKernelType.ADDMM:
            self.computation_kernel = lambda x : torch.addmm(self.bias, x, self.weights)
        elif computation == ComputationKernelType.LINEAR:
            self.computation_kernel = lambda x : self.linear(x)
        elif computation == ComputationKernelType.MATMUL:
            self.computation_kernel = lambda x : torch.matmul(x, self.weights)
        elif computation == ComputationKernelType.MUL:
            self.computation_kernel = lambda x : torch.mul(x, x)

        self.activation_kernel = None
        if activation == ComputationKernelType.RELU:
            self.activation_kernel = F.relu
        elif activation == ComputationKernelType.SIGMOID:
            self.activation_kernel = F.sigmoid
        elif activation == ComputationKernelType.TANH:
            self.activation_kernel = F.tanh

        self.communication_kernel = None
        if communication == CommunicationKernelType.ALLGATHER:
            self.communication_kernel = self.__all_gather_wrapper
        elif communication == CommunicationKernelType.ALLREDUCE:
            self.communication_kernel = self.__all_reduce_wrapper
        elif communication == CommunicationKernelType.ALLTOALL:
            self.communication_kernel = self.__all_to_all_wrapper

        self.step_times = []

    def __all_gather_wrapper(self, x):
        output = torch.empty_like([x.shape[0] * self.num_ranks] + list(x.shape[1:]), device=self.device)
        dist.all_gather_into_tensor(output, x)
        return output

    def __all_reduce_wrapper(self, x):
        dist.all_reduce(x)
        return x

    def __all_to_all_wrapper(self, x):
        output = torch.empty_like(x, device=self.device)
        dist.all_to_all_single(output, x)
        return output

    def forward(self, x):
        for i in range(self.num_layers):
            computation_out = self.computation_kernel(x)
            communication_out = self.communication_kernel(computation_out)
            activation_out = self.activation_out(communication_out)

class DistInferenceSimulation(MicroBenchmark):
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
        self.__device = None
        self.__cuda_available = False

        self.__default_batch_size = 64
        self.__default_input_size = 1024
        self.__default_hidden_size = 1024
        self.__default_num_layers = 10
        self.__default_computation_kernel = ComputationKernelType.MATMUL
        self.__default_communication_kernel = CommunicationKernelType.ALLREDUCE
        self.__default_activation_kernel = ActivationKernelType.RELU
        self.__default_precision = Precision.FLOAT32
        self.__default_num_warmup = 50
        self.__default_num_steps = 200
        self.__default_distributed_impl = DistributedImpl.DDP
        self.__default_distributed_backend = DistributedBackend.NCCL

    def __timer(self):
        """Returns the current time which ensures all previous CUDA events have been finished.

        If there is no GPU present, this defaults to `time.time()`; otherwise it will
        synchronize CUDA before measuring the time.

        Returns:
            Current time in second.
        """
        if self.__cuda_available:
            torch.cuda.synchronize()
        return time.time()

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--batch_size',
            type=int,
            default=self.__default_batch_size,
            required=False,
            help='Batch size.',
        )
        self._parser.add_argument(
            '--input_size',
            type=int,
            default=self.__default_input_size,
            required=False,
            help='Input dimension size.',
        )
        self._parser.add_argument(
            '--hidden_size',
            type=int,
            default=self.__default_hidden_size,
            required=False,
            help='Hidden size.',
        )
        self._parser.add_argument(
            '--num_layers',
            type=int,
            default=self.__default_num_layers,
            required=False,
            help='Number of compute-communicate-activate layers.',
        )
        self._parser.add_argument(
            '--computation_kernel',
            type=ComputationKernelType,
            default=self.__default_computation_kernel,
            required=False,
            help='Computation kernel type. E.g. {}.'.format(' '.join(ComputationKernelType.get_values())),
        )
        self._parser.add_argument(
            '--communication_kernel',
            type=CommunicationKernelType,
            default=self.__default_communication_kernel,
            required=False,
            help='Communication kernel type. E.g. {}.'.format(' '.join(CommunicationKernelType.get_values())),
        )
        self._parser.add_argument(
            '--activation_kernel',
            type=ActivationKernelType,
            default=self.__default_activation_kernel,
            required=False,
            help='Activation kernel type. E.g. {}.'.format(' '.join(ActivationKernelType.get_values())),
        )
        self._parser.add_argument(
            '--precision',
            type=Precision,
            default=self.__default_precision,
            required=False,
            help='Model precision. E.g. {}.'.format(' '.join(Precision.get_values())),
        )
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=self.__default_num_warmup,
            required=False,
            help='Number of warmup steps.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=self.__default_num_steps,
            required=False,
            help='Number of test steps.',
        )
        self._parser.add_argument(
            '--distributed_impl',
            type=DistributedImpl,
            default=self.__default_distributed_impl,
            required=False,
            help='Distributed implementations. E.g. {}.'.format(' '.join(DistributedImpl.get_values())),
        )
        self._parser.add_argument(
            '--distributed_backend',
            type=DistributedBackend,
            default=self.__default_distributed_backend,
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
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)
            torch.distributed.destroy_process_group()
            logger.error('Initialize distributed env failed - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        if torch.cuda.is_available():
            torch.cuda.set_device(self.__local_rank)
            self.__device = torch.device("cuda:{}".format(self.__local_rank))
            self.__cuda_available = True
        else:
            self.__device = torch.device("cpu:{}".format(self.__local_rank))
            self.__cuda_available = False

        return True

    def _benchmark(self):
        """Implementation for benchmarking."""
        batch_size = self._args.batch_size
        input_size = self._args.input_size
        hidden_size = self._args.hidden_size
        num_layers = self._args.num_layers
        computation = self._args.computation_kernel
        communication = self._args.communication_kernel
        activation = self._args.activation_kernel
        num_warmup = self._args.num_warmup
        num_steps = self._args.num_steps

        if self.__local_rank == 0:
            logger.info(
                'Distributed Inference Simulation - using {} GPUs, \
                batch_size={}, input_size={}, hidden_size={}, num_layers={}, \
                computation_kernel={}, communication_kernel={}, activation_kernel={}, precision={}, \
                num_warmup={} num_steps={}'.format(
                    self.__world_size,
                    batch_size, input_size, hidden_size, num_layers,
                    computation, communication, activation, precision,
                    num_warmup, num_steps
                )
            )

        model = DistInferenceSimulationModel(input_size, hidden_size, num_layers, precision, computation, communication, activation, self.__device, self.__world_size)
        data = torch.rand(batch_size, input_size, dtype=getattr(torch, precision.value), device=self.__device)

        # warm up
        warmup_step_times = []
        for i in range(self._args.num_warmup):
            start = self.__timer()
            model.forward(data)
            end = self.__timer()
            warmup_step_times.append((end - start) * 1000)

        # run and collect results
        test_step_times = []
        for i in range(self._args.num_steps):
            start = self.__timer()
            model.forward(data)
            end = self.__timer()
            test_step_times.append((end - start) * 1000)

        if not self._process_numeric_result('warmup_step_times', warmup_step_times):
            return False
        if not self._process_numeric_result('test_step_times', test_step_times):
            return False

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
    'pytorch-dist-inference-simulation', DistInferenceSimulation, parameters=''
)
