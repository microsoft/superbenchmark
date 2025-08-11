# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch CNN models."""

import os
import random
import torch
from torchvision import models

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


def _keep_BatchNorm_as_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        _keep_BatchNorm_as_float(child)
    return module


class PytorchCNN(PytorchBase):
    """The CNN benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._supported_precision = [Precision.FLOAT32, Precision.FLOAT16]
        self._optimizer_type = Optimizer.SGD
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def _enable_deterministic_training(self):
        """Enable deterministic training settings for reproducible results."""
        if hasattr(self._args, 'random_seed'):
            torch.manual_seed(self._args.random_seed)
            random.seed(self._args.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self._args.random_seed)
                torch.cuda.manual_seed_all(self._args.random_seed)

        strict = os.environ.get('SB_STRICT_DETERMINISM', '0') == '1'
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def add_parser_arguments(self):
        """Add the CNN-specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument('--model_type', type=str, required=True, help='The cnn benchmark to run.')
        self._parser.add_argument('--image_size', type=int, default=224, required=False, help='Image size.')
        self._parser.add_argument('--num_classes', type=int, default=1000, required=False, help='Num of class.')
        self._parser.add_argument(
            '--random_seed',
            type=int,
            default=42,
            required=False,
            help='Random seed for deterministic training.'
        )
        self._parser.add_argument(
            '--deterministic',
            action='store_true',
            default=False,
            help='Enable deterministic training for reproducible results.'
        )

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
        if getattr(self._args, 'deterministic', False) and hasattr(self._args, 'random_seed'):
            torch.manual_seed(self._args.random_seed)

        self._dataset = TorchRandomDataset(
            [self._args.sample_count, 3, self._args.image_size, self._args.image_size],
            self._world_size,
            dtype=torch.float32
        )
        if len(self._dataset) == 0:
            logger.error('Generate random dataset failed - model: {}'.format(self._name))
            return False

        return True

    def _create_model(self, precision):
        """Construct the model for benchmarking.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.
        """
        try:
            if getattr(self._args, 'deterministic', False):
                self._enable_deterministic_training()
            self._model = getattr(models, self._args.model_type)()
            self._model = self._model.to(dtype=getattr(torch, precision.value))
            self._model = _keep_BatchNorm_as_float(self._model)
            if self._gpu_available:
                self._model = self._model.cuda()
        except BaseException as e:
            logger.error(
                'Create model with specified precision failed - model: {}, precision: {}, message: {}.'.format(
                    self._name, precision, str(e)
                )
            )
            return False

        if getattr(self._args, 'deterministic', False) and hasattr(self._args, 'random_seed'):
            torch.manual_seed(self._args.random_seed + 1)
        self._target = torch.LongTensor(self._args.batch_size).random_(self._args.num_classes)
        if self._gpu_available:
            self._target = self._target.cuda()

        return True

    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            The step-time list of every training step.
        """
        duration = []
        losses = []
        curr_step = 0
        check_frequency = 100
        while True:
            for idx, sample in enumerate(self._dataloader):
                sample = sample.to(dtype=getattr(torch, precision.value))
                start = self._timer()
                if self._gpu_available:
                    sample = sample.cuda()
                self._optimizer.zero_grad()
                output = self._model(sample)
                # Compute loss in float32 for stability
                loss = self._loss_fn(output.float(), self._target)
                loss.backward()
                self._optimizer.step()
                end = self._timer()
                curr_step += 1
                if curr_step > self._args.num_warmup:
                    # Save the step time of every training/inference step, unit is millisecond.
                    duration.append((end - start) * 1000)
                    try:
                        losses.append(float(loss.detach().item()))
                    except Exception:
                        pass
                    if getattr(self._args, 'deterministic', False) and (curr_step % check_frequency == 0):
                        try:
                            checksum = sum(p.detach().float().sum().item() for p in self._model.parameters())
                            logger.info(f"Checksum at step {curr_step}: {checksum}")
                        except Exception:
                            pass
                    self._log_step_time(curr_step, precision, duration)
                if self._is_finished(curr_step, end, check_frequency):
                    info = {'loss': losses}
                    return (duration, info)

    def _inference_step(self, precision):
        """Define the inference process.

        Args:
            precision (Precision): precision of model and input data,
              such as float32, float16.

        Return:
            The latency list of every inference operation.
        """
        duration = []
        curr_step = 0
        check_frequency = 100
        with torch.no_grad():
            self._model.eval()
            while True:
                for idx, sample in enumerate(self._dataloader):
                    sample = sample.to(dtype=getattr(torch, precision.value))
                    start = self._timer()
                    if self._gpu_available:
                        sample = sample.cuda()
                    self._model(sample)
                    end = self._timer()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every training/inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                        self._log_step_time(curr_step, precision, duration)
                    if self._is_finished(curr_step, end, check_frequency):
                        return duration

    def _process_info(self, model_action, precision, info):
        """Persist extra step-level signals (e.g., loss) into raw_data."""
        try:
            if not info:
                return
            precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'float64': 'fp64', 'bfloat16': 'bf16'}
            prec_value = precision.value if hasattr(precision, 'value') else str(precision)
            prefix = precision_metric.get(prec_value, prec_value)
            metric_loss = f"{prefix}_{model_action}_loss"
            if 'loss' in info and isinstance(info['loss'], list) and len(info['loss']) > 0:
                self._result.add_raw_data(metric_loss, info['loss'], self._args.log_raw_data)
        except Exception:
            pass


# Register CNN benchmarks.
# Reference: https://pytorch.org/vision/0.8/models.html
#            https://github.com/pytorch/vision/tree/v0.8.0/torchvision/models
MODELS = [
    'alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'googlenet', 'inception_v3', 'mnasnet0_5',
    'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11',
    'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
]

for model in MODELS:
    if hasattr(models, model):
        BenchmarkRegistry.register_benchmark('pytorch-' + model, PytorchCNN, parameters='--model_type ' + model)
    else:
        logger.warning('model missing in torchvision.models - model: {}'.format(model))
