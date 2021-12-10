# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the model-benchmark base class."""

import math
import time
import statistics
from abc import abstractmethod

from superbench.common.utils import logger
from superbench.benchmarks import Precision, ModelAction, DistributedImpl, DistributedBackend, BenchmarkType, ReturnCode
from superbench.benchmarks.base import Benchmark
from superbench.benchmarks.context import Enum
from superbench.benchmarks.reducer import ReduceType


class Optimizer(Enum):
    """The Enum class representing different optimizers."""
    SGD = 'sgd'
    ADAM = 'adam'
    ADAMW = 'adamw'


class ModelBenchmark(Benchmark):
    """The base class of E2E model benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._benchmark_type = BenchmarkType.MODEL
        self._world_size = 1
        self._local_rank = None
        self._dataset = None
        self._dataloader = None
        self._model = None
        self._optimizer_type = None
        self._optimizer = None
        self._loss_fn = None
        self._target = None
        self._supported_precision = []
        self._gpu_available = None

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

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
            '--sample_count',
            type=int,
            default=1024,
            required=False,
            help='The number of data samples in dataset.',
        )
        self._parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            required=False,
            help='The number of batch size.',
        )
        self._parser.add_argument(
            '--precision',
            type=Precision,
            default=[Precision.FLOAT32, Precision.FLOAT16],
            nargs='+',
            required=False,
            help='Model precision. E.g. {}.'.format(' '.join(Precision.get_values())),
        )
        self._parser.add_argument(
            '--model_action',
            type=ModelAction,
            default=[ModelAction.TRAIN],
            nargs='+',
            required=False,
            help='Benchmark model process. E.g. {}.'.format(' '.join(ModelAction.get_values())),
        )
        self._parser.add_argument(
            '--distributed_impl',
            type=DistributedImpl,
            default=None,
            required=False,
            help='Distributed implementations. E.g. {}.'.format(' '.join(DistributedImpl.get_values())),
        )

        self._parser.add_argument(
            '--distributed_backend',
            type=DistributedBackend,
            default=None,
            required=False,
            help='Distributed backends. E.g. {}.'.format(' '.join(DistributedBackend.get_values())),
        )

        self._parser.add_argument(
            '--no_gpu',
            action='store_true',
            default=False,
            help='Disable GPU training.',
        )

        self._parser.add_argument(
            '--pin_memory',
            action='store_true',
            default=False,
            help='Enable option to pin memory in data loader.',
        )

        self._parser.add_argument(
            '--force_fp32',
            action='store_true',
            default=False,
            help='Enable option to use full float32 precision.',
        )

    @abstractmethod
    def _judge_gpu_availability(self):
        """Judge GPUs' availability according to arguments and running environment."""
        pass

    @abstractmethod
    def _set_force_fp32(self):
        """Set the config that controls whether full float32 precision will be used.

        On Ampere or newer GPUs, pytorch and tensorflow will use TF32 instead of FP32 by default.
        We can disable TF32 execution by setting force_fp32 as True.
        """
        pass

    @abstractmethod
    def _init_distributed_setting(self):
        """Initialize the distributed library and bind the worker to GPU.

        Return:
            True if distributed library is initialized successfully.
        """
        pass

    @abstractmethod
    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
        pass

    @abstractmethod
    def _init_dataloader(self):
        """Initialize the dataloader.

        Return:
            True if dataloader is created successfully.
        """
        pass

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self._judge_gpu_availability()
        self._set_force_fp32()
        logger.info(
            'Model placement - model: {}, GPU availablility: {}, pin memory: {}, force fp32: {}.'.format(
                self._name, self._gpu_available, self._args.pin_memory, self._args.force_fp32
            )
        )

        if not self._init_distributed_setting():
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_INIT_FAILURE)
            return False

        # Set sample_count aligned with batch_size.
        self._args.sample_count = math.ceil(self._args.sample_count / self._args.batch_size) * self._args.batch_size

        if not self._generate_dataset():
            self._result.set_return_code(ReturnCode.DATASET_GENERATION_FAILURE)
            return False

        if not self._init_dataloader():
            self._result.set_return_code(ReturnCode.DATALOADER_INIT_FAILURE)
            return False

        return True

    @abstractmethod
    def _create_optimizer(self):
        """Create the optimzier instance used for training and wrap with distributed library if need.

        Return:
            True if optimizer instance is created successfully.
        """
        pass

    @abstractmethod
    def _create_model(self, precision):
        """Construct the model for benchmarking.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.
        """
        pass

    def __train(self, precision):
        """Launch the training benchmark.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            True if step_times list is not empty.
        """
        if not self._create_model(precision):
            self._result.set_return_code(ReturnCode.MODEL_CREATION_FAILURE)
            return False

        if not self._create_optimizer():
            self._result.set_return_code(ReturnCode.OPTIMIZER_CREATION_FAILURE)
            return False

        # The unit of step time should be millisecond.
        step_times = self._train_step(precision)
        if not self.__process_model_result(ModelAction.TRAIN, precision, step_times):
            self._result.set_return_code(ReturnCode.INVALID_BENCHMARK_RESULT)
            return False

        logger.info(
            'Average train time - round: {}, model: {}, precision: {}, step time: {:.6f} ms.'.format(
                self._curr_run_index, self._name, precision, statistics.mean(step_times)
            )
        )

        return True

    def __inference(self, precision):
        """Launch the inference benchmark.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            True if step_times list is not empty.
        """
        self._create_model(precision)
        # The unit of step time should be millisecond.
        step_times = self._inference_step(precision)
        if not self.__process_model_result(ModelAction.INFERENCE, precision, step_times):
            self._result.set_return_code(ReturnCode.INVALID_BENCHMARK_RESULT)
            return False

        logger.info(
            'Average inference time - round: {}, model: {}, precision: {}, step time: {:.6f} ms.'.format(
                self._curr_run_index, self._name, precision, statistics.mean(step_times)
            )
        )

        return True

    @abstractmethod
    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            The step-time list of every training step.
        """
        pass

    @abstractmethod
    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (Precision): precision of model and input data,
              such as float32, float16.

        Return:
            The latency list of every inference operation.
        """
        pass

    def _benchmark(self):
        """Implementation for benchmarking.

        Return:
            True if run benchmark successfully.
        """
        precision_need_to_run = list()
        for precision in self._args.precision:
            # Check if the precision is supported or not.
            if precision not in self._supported_precision:
                logger.warning(
                    'Can not run with specified precision - model: {}, supprted precision: {}, specified precision: {}'.
                    format(self._name, ' '.join([p.value for p in self._supported_precision]), precision)
                )
            else:
                precision_need_to_run.append(precision)

        if len(precision_need_to_run) == 0:
            self._result.set_return_code(ReturnCode.NO_SUPPORTED_PRECISION)
            return False

        for precision in precision_need_to_run:
            for model_action in self._args.model_action:
                self._sub_benchmark_start_time = time.time()
                if model_action == ModelAction.TRAIN:
                    if not self.__train(precision):
                        return False
                elif model_action == ModelAction.INFERENCE:
                    if not self.__inference(precision):
                        return False
                else:
                    logger.warning(
                        'Model action has no implementation yet - model: {}, model_action: {}'.format(
                            self._name, model_action
                        )
                    )

        return True

    def _is_finished(self, curr_step, curr_time):
        total_steps = self._args.num_warmup + self._args.num_steps

        if (
            (self._args.duration > 0 and (curr_time - self._sub_benchmark_start_time) >= self._args.duration)
            or (total_steps > 0 and curr_step >= total_steps)
        ):
            return True

        return False

    def __process_model_result(self, model_action, precision, step_times):
        """Function to process raw results and save the summarized results.

        Args:
            model_action (ModelAction): train or inference.
            precision (Precision): precision of model and input data, such as float32, float16.
            step_times (list): The step time list of every training/inference step, unit is millisecond.

        Return:
            True if step_times list is not empty.
        """
        if len(step_times) == 0:
            logger.error(
                'Step time list is empty - round: {}, model: {}, model_action: {}, precision: {}.'.format(
                    self._curr_run_index, self._name, model_action, precision
                )
            )
            return False

        precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'float64': 'fp64', 'bfloat16': 'bf16'}
        if precision.value in precision_metric.keys():
            precision = precision_metric[precision.value]
        metric = '{}_{}_step_time'.format(precision, model_action)
        self._result.add_raw_data(metric, step_times)
        avg = statistics.mean(step_times)
        self._result.add_result(metric, avg, reduce_type=ReduceType.MAX if model_action is ModelAction.TRAIN else None)

        # The unit of step time is millisecond, use it to calculate the throughput with the unit samples/sec.
        millisecond_per_second = 1000
        throughput = [millisecond_per_second / step_time * self._args.batch_size for step_time in step_times]
        metric = '{}_{}_throughput'.format(precision, model_action)
        self._result.add_raw_data(metric, throughput)
        avg = statistics.mean(throughput)
        self._result.add_result(metric, avg, reduce_type=ReduceType.MIN if model_action is ModelAction.TRAIN else None)

        return True

    @abstractmethod
    def _cal_params_count(self):
        """Calculate the parameters scale of the model.

        Return:
            The count of trainable parameters.
        """
        pass

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass
