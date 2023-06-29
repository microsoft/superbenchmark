# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the distributed inference benchmark."""

import os
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist

from superbench.common.utils import logger
from superbench.benchmarks import DistributedImpl, DistributedBackend, BenchmarkRegistry, ReturnCode, Precision
from superbench.benchmarks.micro_benchmarks import MicroBenchmark
from superbench.benchmarks.context import Enum
from superbench.benchmarks.reducer import ReduceType


class ComputationKernelType(Enum):
    """The Enum class representing different computation kernel type."""
    ADDMM = 'addmm'
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


class DistInferenceModel(torch.nn.Module):
    """The model class for distributed inference benchmark."""
    def __init__(
        self, input_size, hidden_size, num_layers, computation, communication, activation, precision, num_ranks, device
    ):
        """Constructor.

        Args:
            input_size (int): input data dimension.
            hidden_size (int): hidden layer dimension.
            num_layers (int): number of layers in the model.
            computation (ComputationKernelType): type of computation kernel of this model.
            communication (CommunicationKernelType): type of communication kernel of this model.
            activation (ActivationKernelType): type of activation kernel of this model.
            precision (Precision): data type of this model.
            num_ranks (int): number of ranks in this model runs.
            device (torch.device): device this model runs on.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.weights = torch.rand(
            self.input_size, self.hidden_size, dtype=getattr(torch, precision.value), device=device
        )
        self.bias = torch.rand(self.hidden_size, dtype=getattr(torch, precision.value), device=device)
        self.num_ranks = num_ranks
        self.step_times = []

        self.__init_computation_kernels(computation)
        self.__init_communication_kernels(communication)
        self.__init_activation_kernels(activation)

    def __init_computation_kernels(self, computation):
        """Select computation kernel according to option.

        Args:
            computation (ComputationKernelType): the type of the computation kernel to run.
        """
        self.computation_kernel = None
        if computation == ComputationKernelType.ADDMM:
            self.computation_kernel = lambda x: torch.addmm(self.bias, x, self.weights)
        elif computation == ComputationKernelType.MATMUL:
            self.computation_kernel = lambda x: torch.matmul(x, self.weights)
        elif computation == ComputationKernelType.MUL:
            self.computation_kernel = lambda x: torch.mul(x, x)

    def __init_communication_kernels(self, communication):
        """Select communication kernel according to option.

        Args:
            communication (CommunicationKernelType): the type of the communication kernel to run.
        """
        self.communication_kernel = None
        if communication == CommunicationKernelType.ALLGATHER:
            self.communication_kernel = self.__all_gather_wrapper
        elif communication == CommunicationKernelType.ALLREDUCE:
            self.communication_kernel = self.__all_reduce_wrapper
        elif communication == CommunicationKernelType.ALLTOALL:
            self.communication_kernel = self.__all_to_all_wrapper

    def __init_activation_kernels(self, activation):
        """Select activation kernel according to option.

        Args:
            activation (ActivationKernelType): the type of the activation kernel to run.
        """
        self.activation_kernel = None
        if activation == ActivationKernelType.RELU:
            self.activation_kernel = F.relu
        elif activation == ActivationKernelType.SIGMOID:
            self.activation_kernel = F.sigmoid
        elif activation == ActivationKernelType.TANH:
            self.activation_kernel = F.tanh

    def __all_gather_wrapper(self, x):
        """All-gather wrapper with output initialization.

        Args:
            x (Tensor): input.

        Return:
            Tensor after all-gather.
        """
        output = torch.empty([x.shape[0] * self.num_ranks] + list(x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(output, x)
        return output

    def __all_reduce_wrapper(self, x):
        """All-reduce wrapper.

        Args:
            x (Tensor): input.

        Return:
            Tensor after all-reduce.
        """
        dist.all_reduce(x)
        return x

    def __all_to_all_wrapper(self, x):
        """All-to-all wrapper with output initialization.

        Args:
            x (Tensor): input.

        Return:
            Tensor after all-to-all.
        """
        output = torch.empty_like(x)
        dist.all_to_all_single(output, x)
        return output

    def forward(self, x):
        """Do forward loops.

        Args:
            x (Tensor): input.

        Return:
            Tensor after the whole inference process.
        """
        activation_out = None
        for i in range(self.num_layers):
            computation_out = self.computation_kernel(x)
            communication_out = self.communication_kernel(computation_out)
            activation_out = self.activation_kernel(communication_out)
        return activation_out


class DistInference(MicroBenchmark):
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

    def __timer(self):
        """Returns the current time which ensures all previous CUDA events have been finished.

        If there is no GPU present, this defaults to `time.time()`; otherwise it will
        synchronize CUDA before measuring the time.

        Return:
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
            default=64,
            required=False,
            help='Batch size.',
        )
        self._parser.add_argument(
            '--input_size',
            type=int,
            default=1024,
            required=False,
            help='Input dimension size.',
        )
        self._parser.add_argument(
            '--hidden_size',
            type=int,
            default=1024,
            required=False,
            help='Hidden size.',
        )
        self._parser.add_argument(
            '--num_layers',
            type=int,
            default=1,
            required=False,
            help='Number of compute-communicate-activate layers.',
        )
        self._parser.add_argument(
            '--computation_kernel',
            type=ComputationKernelType,
            default=ComputationKernelType.MATMUL,
            required=False,
            help='Computation kernel type. E.g. {}.'.format(' '.join(ComputationKernelType.get_values())),
        )
        self._parser.add_argument(
            '--communication_kernel',
            type=CommunicationKernelType,
            default=CommunicationKernelType.ALLREDUCE,
            required=False,
            help='Communication kernel type. E.g. {}.'.format(' '.join(CommunicationKernelType.get_values())),
        )
        self._parser.add_argument(
            '--activation_kernel',
            type=ActivationKernelType,
            default=ActivationKernelType.RELU,
            required=False,
            help='Activation kernel type. E.g. {}.'.format(' '.join(ActivationKernelType.get_values())),
        )
        self._parser.add_argument(
            '--precision',
            type=Precision,
            default=Precision.FLOAT32,
            required=False,
            help='Model precision. E.g. {}.'.format(' '.join(Precision.get_values())),
        )
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=50,
            required=False,
            help='Number of warmup steps.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=10000,
            required=False,
            help='Number of test steps.',
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
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)
            torch.distributed.destroy_process_group()
            logger.error('Initialize distributed env failed - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        if torch.cuda.is_available():
            torch.cuda.set_device(self.__local_rank)
            self.__device = torch.device('cuda:{}'.format(self.__local_rank))
            self.__cuda_available = True
        else:
            self.__device = torch.device('cpu:{}'.format(self.__local_rank))
            self.__cuda_available = False

        return True

    def _prepare_model(
        self, input_size, hidden_size, num_layers, computation, communication, activation, precision, num_ranks
    ):
        """Prepare model.

        Args:
            input_size (int): input data dimension.
            hidden_size (int): hidden layer dimension.
            num_layers (int): number of layers in the model.
            computation (ComputationKernelType): type of computation kernel of this model.
            communication (CommunicationKernelType): type of communication kernel of this model.
            activation (ActivationKernelType): type of activation kernel of this model.
            precision (Precision): data type of this model.
            num_ranks (int): number of ranks in this model runs.

        Return:
            Model prepared.
        """
        model = DistInferenceModel(
            input_size, hidden_size, num_layers, computation, communication, activation, precision, num_ranks,
            self.__device
        )
        model = model.to(dtype=getattr(torch, precision.value))
        if self.__cuda_available:
            model = model.cuda()
        return model

    def _run_model(self, model, batch_size, input_size, precision, device, num_warmup, num_steps):
        """Run model and collect step times.

        Args:
            model (torch.nn.Module): model to run.
            batch_size (int): batch size of input data.
            input_size (int): input data dimension.
            precision (Precision): data type of this model.
            device (torch.device): device this model runs on.
            num_warmup (int): number of warm-up runs.
            num_steps (int): number of test runs.

        Return:
            Model step times collected.
        """
        data = torch.rand(batch_size, input_size, dtype=getattr(torch, precision.value), device=self.__device)

        # warm up
        for i in range(num_warmup):
            model(data)

        # run and collect results
        step_times = [0.] * num_steps
        for i in range(self._args.num_steps):
            start = self.__timer()
            model(data)
            end = self.__timer()
            step_times[i] = (end - start) * 1000

        return step_times

    def _process_data(self, step_times):
        """Process data.

        Args:
            step_times (List[float]): Model step times collected.

        Return:
            True if _process_data succeeds.
        """
        if not self._process_numeric_result('step_times', step_times, reduce_type=ReduceType.MAX, cal_percentile=True):
            return False
        return True

    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if _benchmark succeeds.
        """
        batch_size = self._args.batch_size
        input_size = self._args.input_size
        hidden_size = self._args.hidden_size
        num_layers = self._args.num_layers
        computation = self._args.computation_kernel
        communication = self._args.communication_kernel
        activation = self._args.activation_kernel
        precision = self._args.precision
        num_warmup = self._args.num_warmup
        num_steps = self._args.num_steps

        if self.__local_rank == 0:
            logger.info(
                'Distributed Inference - using {} GPUs: '
                'batch_size={}, input_size={}, hidden_size={}, num_layers={}, '
                'computation_kernel={}, communication_kernel={}, activation_kernel={}, precision={}, '
                'num_warmup={} num_steps={}'.format(
                    self.__world_size, batch_size, input_size, hidden_size, num_layers, computation, communication,
                    activation, precision, num_warmup, num_steps
                )
            )

        # Prepare model
        model = self._prepare_model(
            input_size, hidden_size, num_layers, computation, communication, activation, precision, self.__world_size
        )

        # Run model
        step_times = self._run_model(model, batch_size, input_size, precision, self.__device, num_warmup, num_steps)

        # Process data and return
        return self._process_data(step_times)

    def _postprocess(self):
        """Postprocess/cleanup operations after the benchmarking.

        Return:
            True if _postprocess succeeds.
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


BenchmarkRegistry.register_benchmark('pytorch-dist-inference', DistInference, parameters='')
