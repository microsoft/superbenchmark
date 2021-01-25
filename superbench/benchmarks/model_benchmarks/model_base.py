# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import abstractmethod
from superbench.common.utils import logger
from superbench.benchmarks.benchmark_base import Benchmark


class ModelBenchmark(Benchmark):
    '''The base class of E2E model benchmarks.

    Args:
        name: benchmark name.
        argv: benchmark parameters.
    '''
    def __init__(self, name, argv=''):
        super().__init__(name, argv)

        self._world_size = 1
        self._dataset = None
        self._dataloader = None
        self._model = None
        self._optimizer = None
        self._loss_fn = None
        self._target = None
        self._supported_precision = []

    def add_parser_auguments(self):
        super().add_parser_auguments()

        self._parser.add_argument(
            '--num_warmup_steps', type=int, default=8,
            required=False, help='The number of warmup step'
        )
        self._parser.add_argument(
            '--num_steps', type=int, default=256,
            required=False, help='The number of test step'
        )
        self._parser.add_argument(
            '--batch_size', type=int, default=32,
            required=False, help='The number of batch size'
        )
        self._parser.add_argument(
            '--precision', type=str, default='float,half',
            required=False, help='Model precision, such as (float, half).'
        )
        self._parser.add_argument(
            '--model_action', type=str, default='train',
            required=False, help='Benchmark type, train or inference.'
        )
        self._parser.add_argument(
            '--distributed_mode', type=str, default='default',
            required=False, help='Distributed mode. E.g. horovod, native.'
        )

    def parse_args(self):
        self._args, unknown = super().parse_args()
        self._args.precision = [item.strip()
                                for item in self._args.precision.split(',')]
        return self._args, unknown

    @abstractmethod
    def init_distributed_setting(self):
        pass

    @abstractmethod
    def generate_dataset(self):
        pass

    @abstractmethod
    def init_dataloader(self):
        pass

    def preprocess(self):
        super().preprocess()
        self.init_distributed_setting()
        self.generate_dataset()
        self.init_dataloader()

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    def train(self, precision):
        self.create_model(precision)
        self.create_optimizer()
        step_times = self.training_step(precision)
        logger.info('{} model {} average train time: {} ms'.format(
            self._name, precision, sum(step_times) / len(step_times)))

        self.process_result('train', precision, step_times)

    def inference(self, precision):
        self.create_model(precision)
        step_times = self.inference_step(precision)
        logger.info('{} model {} average inference time: {} ms'.format(
            self._name, precision, sum(step_times) / len(step_times)))

        self.process_result('inference', precision, step_times)

    @abstractmethod
    def training_step(self, precision):
        pass

    @abstractmethod
    def inference_step(self, precision):
        pass

    def benchmarking(self):
        for precision in self._args.precision:
            if precision not in self._supported_precision:
                logger.warning('{} model can\'t run with precision {}'.format(
                    self._name, precision))
                continue
            if self._args.model_action == 'train':
                self.train(precision)
            elif self._args.model_action == 'inference':
                self.inference(precision)
            else:
                logger.warning('{} model has unknown model action {}'.format(
                    self._name, self._args.model_action))

    def process_result(self, model_action, precision, step_times):
        metric = 'steptime_{}_{}'.format(model_action, precision)
        self._result.add_raw_data(metric, step_times)
        avg = sum(step_times) / len(step_times)
        self._result.add_result(metric, avg)

        metric = 'throughput_{}_{}'.format(model_action, precision)
        throughput = [1000 / step_time *
                      self._args.batch_size for step_time in step_times]
        self._result.add_raw_data(metric, throughput)
        avg = sum(throughput) / len(throughput)
        self._result.add_result(metric, avg)

    @abstractmethod
    def cal_params_size(self):
        pass

    @abstractmethod
    def print_env_info(self):
        pass
