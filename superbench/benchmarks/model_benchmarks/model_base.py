# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the model-benchmark base class."""

from abc import abstractmethod

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkType, BenchmarkResult
from superbench.benchmarks.base import Benchmark


class ModelBenchmark(Benchmark):
    """The base class of E2E model benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)

        self._world_size = 1
        self._dataset = None
        self._dataloader = None
        self._model = None
        self._optimizer = None
        self._loss_fn = None
        self._target = None
        self._supported_precision = []

    def add_parser_auguments(self):
        """Add the specified auguments."""
        super().add_parser_auguments()

        self._parser.add_argument(
            '--num_warmup_steps',
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
        self._parser.add_argument(
            '--precision',
            type=str,
            default='float,half',
            required=False,
            help='Model precision, such as (float, half).',
        )
        self._parser.add_argument(
            '--model_action',
            type=str,
            default='train',
            required=False,
            help='Benchmark type, train or inference.',
        )
        self._parser.add_argument(
            '--distributed_mode',
            type=str,
            default='default',
            required=False,
            help='Distributed mode. E.g. horovod, native.',
        )

    def parse_args(self):
        """Parse the arguments.

        Return:
            The parsed arguments and unknown arguments.
        """
        self._args, unknown = super().parse_args()
        self._args.precision = [item.strip() for item in self._args.precision.split(',')]
        return self._args, unknown

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
        self._result = BenchmarkResult(self._name, BenchmarkType.MODEL.value, run_count=self._args.run_count)
        self._init_distributed_setting()
        self._generate_dataset()
        self._init_dataloader()

    @abstractmethod
    def _create_optimizer(self):
        """Create the optimzier instance used for training."""
        pass

    @abstractmethod
    def _create_model(self):
        """Construct the model for benchmarking."""
        pass

    def __train(self, precision):
        """Launch the training benchmark.

        Args:
            precision (str): precision of model and input data,
              such as float, half.
        """
        self._create_model(precision)
        self._create_optimizer()
        step_times = self._training_step(precision)
        logger.info(
            '{} model {} average train time: {} ms'.format(self._name, precision,
                                                           sum(step_times) / len(step_times))
        )

        self.__process_result('train', precision, step_times)

    def __inference(self, precision):
        """Launch the inference benchmark."""
        self._create_model(precision)
        step_times = self._inference_step(precision)
        logger.info(
            '{} model {} average inference time: {} ms'.format(
                self._name, precision,
                sum(step_times) / len(step_times)
            )
        )

        self.__process_result('inference', precision, step_times)

    @abstractmethod
    def _training_step(self, precision):
        """Define the training process.

        Args:
            precision (str): precision of model and input data,
              such as float, half.

        Return:
            The step-time list of every training step.
        """
        pass

    @abstractmethod
    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (str): precision of model and input data,
              such as float, half.

        Return:
            The latency list of every inference operation.
        """
        pass

    def _benchmarking(self):
        """Implementation for benchmarking."""
        for precision in self._args.precision:
            # Check if the precision is supported or not.
            if precision not in self._supported_precision:
                logger.warning("{} model can't run with precision {}".format(self._name, precision))
                continue

            if self._args.model_action == 'train':
                self.__train(precision)
            elif self._args.model_action == 'inference':
                self.__inference(precision)
            else:
                logger.warning('{} model has unknown model action {}'.format(self._name, self._args.model_action))

    def __process_result(self, model_action, precision, step_times):
        """Function to process raw results and save the summarized results.

        Args:
            model_action (str): train or inference.
            precision (str): precision of model and input data,
              such as float, half.
            step_times (list): The list of every training/inference step.
        """
        metric = 'steptime_{}_{}'.format(model_action, precision)
        self._result.add_raw_data(metric, step_times)
        avg = sum(step_times) / len(step_times)
        self._result.add_result(metric, avg)

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
        pass
