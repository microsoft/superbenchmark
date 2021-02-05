# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the model-benchmark base class."""

from abc import abstractmethod
from enum import Enum

from superbench.common.utils import logger
from superbench.benchmarks import Precision, ModelAction, BenchmarkType
from superbench.benchmarks.base import Benchmark


class DistributedMode(Enum):
    """The Enum class representing different distributed mode."""
    DDP = 'pytorch-ddp'
    HOROVOD = 'horovod'


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
        self._world_size = None
        self._dataset = None
        self._dataloader = None
        self._model = None
        self._optimizer = None
        self._loss_fn = None
        self._target = None
        self._supported_precision = []

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=64,
            metavar='',
            required=False,
            help='The number of warmup step',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=2048,
            metavar='',
            required=False,
            help='The number of test step',
        )
        self._parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            metavar='',
            required=False,
            help='The number of batch size',
        )
        precision_choice = [p.value for p in Precision]
        self._parser.add_argument(
            '--precision',
            type=str,
            default=[Precision.FLOAT32.value, Precision.FLOAT16.value],
            choices=precision_choice,
            nargs='+',
            metavar='',
            required=False,
            help='Model precision. E.g. {}.'.format(' '.join(precision_choice)),
        )
        model_action_choice = [p.value for p in ModelAction]
        self._parser.add_argument(
            '--model_action',
            type=str,
            default=[ModelAction.TRAIN.value],
            choices=model_action_choice,
            nargs='+',
            metavar='',
            required=False,
            help='Benchmark type. E.g. {}.'.format(' '.join(model_action_choice)),
        )
        distributed_mode_choice = [p.value for p in DistributedMode]
        self._parser.add_argument(
            '--distributed_mode',
            type=str,
            choices=distributed_mode_choice,
            metavar='',
            required=False,
            help='Distributed mode. E.g. {}'.format(' '.join(distributed_mode_choice)),
        )

    '''
    def parse_args(self):
        """Parse the arguments.

        Return:
            The parsed arguments and unknown arguments.
        """
        self._args, unknown = super().parse_args()
        self._args.precision = [item.strip() for item in self._args.precision.split(',')]
        return self._args, unknown
    '''

    @abstractmethod
    def _init_distributed_setting(self):
        """Initialize the distributed library and bind the worker to GPU."""
        pass

    @abstractmethod
    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info."""
        pass

    @abstractmethod
    def _init_dataloader(self):
        """Initialize the distributed dataloader."""
        pass

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking."""
        super()._preprocess()
        self._init_distributed_setting()
        self._generate_dataset()
        self._init_dataloader()

    @abstractmethod
    def _create_optimizer(self):
        """Create the optimzier instance used for training."""
        pass

    @abstractmethod
    def _create_model(self, precision):
        """Construct the model for benchmarking.

        Args:
            precision (str): precision of model and input data, such as float32, float16.
        """
        pass

    def __train(self, precision):
        """Launch the training benchmark.

        Args:
            precision (str): precision of model and input data, such as float32, float16.
        """
        self._create_model(precision)
        self._create_optimizer()
        # The unit of step time should be milisecond.
        step_times = self._train_step(precision)
        logger.info(
            'Average train time - round: {}, model: {}, precision: {}, step time: {} ms.'.format(
                self._curr_index, self._name, precision,
                sum(step_times) / len(step_times)
            )
        )

        self.__process_model_result(ModelAction.TRAIN.value, precision, step_times)

    def __inference(self, precision):
        """Launch the inference benchmark.

        Args:
            precision (str): precision of model and input data, such as float32, float16.
        """
        self._create_model(precision)
        # The unit of step time should be milisecond.
        step_times = self._inference_step(precision)
        logger.info(
            'Average inference time - round: {}, model: {}, precision: {}, step time: {} ms.'.format(
                self._curr_index, self._name, precision,
                sum(step_times) / len(step_times)
            )
        )

        self.__process_model_result(ModelAction.INFEENCE.value, precision, step_times)

    @abstractmethod
    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (str): precision of model and input data, such as float32, float16.

        Return:
            The step-time list of every training step.
        """
        pass

    @abstractmethod
    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (str): precision of model and input data,
              such as float32, float16.

        Return:
            The latency list of every inference operation.
        """
        pass

    def _benchmark(self):
        """Implementation for benchmarking."""
        for precision in self._args.precision:
            # Check if the precision is supported or not.
            if precision not in self._supported_precision:
                logger.warning("{} model can't run with precision {}".format(self._name, precision))
                continue

            for model_action in self._args.model_action:
                if model_action == ModelAction.TRAIN.value:
                    self.__train(precision)
                elif model_action == ModelAction.INFEENCE.value:
                    self.__inference(precision)
                else:
                    logger.warning('{} model has unknown model action {}'.format(self._name, self._args.model_action))

    def __process_model_result(self, model_action, precision, step_times):
        """Function to process raw results and save the summarized results.

        Args:
            model_action (str): train or inference.
            precision (str): precision of model and input data, such as float32, float16.
            step_times (list): The step time list of every training/inference step, unit is millisecond.
        """
        metric = 'steptime_{}_{}'.format(model_action, precision)
        self._result.add_raw_data(metric, step_times)
        avg = sum(step_times) / len(step_times)
        self._result.add_result(metric, avg)

        # The unit of step time is milisecond, use it to calculate the throughput with the unit samples/sec.
        metric = 'throughput_{}_{}'.format(model_action, precision)
        throughput = [1000 / step_time * self._args.batch_size for step_time in step_times]
        self._result.add_raw_data(metric, throughput)
        avg = sum(throughput) / len(throughput)
        self._result.add_result(metric, avg)

    @abstractmethod
    def _cal_params_size(self):
        """Calculate the parameters scale of the model.

        Return:
            The count of trainable parameters.
        """
        pass

    def print_env_info(self):
        """Print environments or dependencies information."""
        # TODO: will implement it when add real benchmarks in the future.
        pass
