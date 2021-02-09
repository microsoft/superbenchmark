# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the model-benchmark base class."""

from abc import abstractmethod

from superbench.common.utils import logger
from superbench.benchmarks import Enum, Precision, ModelAction, BenchmarkType, ReturnCode
from superbench.benchmarks.base import Benchmark


class DistributedMode(Enum):
    """The Enum class representing different distributed mode."""
    PYTORCH_DDP_NCCL = 'pytorch-ddp-nccl'
    TF_MIRRORED = 'tf-mirrored'
    TF_MW_MIRRORED = 'tf-multiworkermirrored'
    TF_PS = 'tf-parameterserver'
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
            required=False,
            help='The number of warmup step',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=2048,
            required=False,
            help='The number of test step',
        )
        self._parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            required=False,
            help='The number of batch size',
        )
        self._parser.add_argument(
            '--precision',
            type=str,
            default=[Precision.FLOAT32.value, Precision.FLOAT16.value],
            nargs='+',
            required=False,
            help='Model precision. E.g. {}.'.format(' '.join(Precision.get_values())),
        )
        self._parser.add_argument(
            '--model_action',
            type=str,
            default=[ModelAction.TRAIN.value],
            nargs='+',
            required=False,
            help='Benchmark model process. E.g. {}.'.format(' '.join(ModelAction.get_values())),
        )
        self._parser.add_argument(
            '--distributed_mode',
            type=str,
            default=None,
            required=False,
            help='Distributed mode. E.g. {}'.format(' '.join(DistributedMode.get_values())),
        )

    def parse_args(self):
        """Parse and check the validation of arguments.

        Return:
            ret (bool): whether parse succeed or not.
            args (argparse.Namespace): parsed arguments.
            unknown (list): unknown arguments.
        """
        ret, args, unknown = super().parse_args()
        for precision in args.precision:
            if precision not in Precision.get_values():
                logger.error(
                    'Invalid precision argument - benchmark: {}, expect: {}, got: {}.'.format(
                        self._name, ' '.join(Precision.get_values()), precision
                    )
                )
                ret = False
        for model_action in args.model_action:
            if model_action not in ModelAction.get_values():
                logger.error(
                    'Invalid model_action argument - benchmark: {}, expect: {}, got: {}.'.format(
                        self._name, ' '.join(ModelAction.get_values()), model_action
                    )
                )
                ret = False

        if args.distributed_mode:
            for distributed_mode in args.distributed_mode:
                if distributed_mode not in DistributedMode.get_values():
                    logger.error(
                        'Invalid distributed_mode argument - benchmark: {}, expect: {}, got: {}.'.format(
                            self._name, ' '.join(DistributedMode.get_values()), distributed_mode
                        )
                    )
                    ret = False

        return ret, args, unknown

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
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        ret = super()._preprocess()
        if not ret:
            return False

        self._init_distributed_setting()
        self._generate_dataset()
        self._init_dataloader()
        return True

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

        Return:
            True if step_times list is not empty.
        """
        self._create_model(precision)
        self._create_optimizer()
        # The unit of step time should be millisecond.
        step_times = self._train_step(precision)
        if len(step_times) == 0:
            logger.error(
                'Step time list for training is empty - round: {}, model: {}, precision: {}.'.format(
                    self._curr_run_index, self._name, precision
                )
            )
            return False

        average_time = sum(step_times) / len(step_times)
        logger.info(
            'Average train time - round: {}, model: {}, precision: {}, step time: {:.6f} ms.'.format(
                self._curr_run_index, self._name, precision, average_time
            )
        )

        self.__process_model_result(ModelAction.TRAIN.value, precision, step_times)
        return True

    def __inference(self, precision):
        """Launch the inference benchmark.

        Args:
            precision (str): precision of model and input data, such as float32, float16.

        Return:
            True if step_times list is not empty.
        """
        self._create_model(precision)
        # The unit of step time should be millisecond.
        step_times = self._inference_step(precision)
        if len(step_times) == 0:
            logger.error(
                'Step time list for inference is empty - round: {}, model: {}, precision: {}.'.format(
                    self._curr_run_index, self._name, precision
                )
            )
            return False

        average_time = sum(step_times) / len(step_times)
        logger.info(
            'Average inference time - round: {}, model: {}, precision: {}, step time: {:.6f} ms.'.format(
                self._curr_run_index, self._name, precision, average_time
            )
        )

        self.__process_model_result(ModelAction.INFERENCE.value, precision, step_times)
        return True

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
                    format(self._name, ' '.join(self._supported_precision), precision)
                )
            else:
                precision_need_to_run.append(precision)

        if len(precision_need_to_run) == 0:
            self._result.set_return_code(ReturnCode.NO_SUPPORTED_PRECISION.value)
            return False

        for precision in precision_need_to_run:
            for model_action in self._args.model_action:
                if model_action == ModelAction.TRAIN.value:
                    if not self.__train(precision):
                        self._result.set_return_code(ReturnCode.MODEL_TRAIN_FAILURE.value)
                        return False
                elif model_action == ModelAction.INFERENCE.value:
                    if not self.__inference(precision):
                        self._result.set_return_code(ReturnCode.MODEL_INFERENCE_FAILURE.value)
                        return False
                else:
                    logger.warning(
                        'Model action has no implementation yet - model: {}, model_action: {}'.format(
                            self._name, model_action
                        )
                    )

        return True

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

        # The unit of step time is millisecond, use it to calculate the throughput with the unit samples/sec.
        millisecond_per_second = 1000
        throughput = [millisecond_per_second / step_time * self._args.batch_size for step_time in step_times]
        metric = 'throughput_{}_{}'.format(model_action, precision)
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
