# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch CNN models."""

import time

import torch
from torchvision import models

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


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

    def add_parser_arguments(self):
        """Add the CNN-specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument('--model_type', type=str, required=True, help='The cnn benchmark to run.')
        self._parser.add_argument('--image_size', type=int, default=224, required=False, help='Image size.')
        self._parser.add_argument('--num_classes', type=int, default=100, required=False, help='Num of class.')

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
        self._dataset = TorchRandomDataset(
            [self._args.sample_count, 3, self._args.image_size, self._args.image_size],
            self._world_size,
            dtype=torch.long
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
            self._model = getattr(models, self._args.model_type)()
            self._model = self._model.to(dtype=getattr(torch, precision.value))
            if self._gpu_available:
                self._model = self._model.cuda()
        except BaseException as e:
            logger.error(
                'Create model with specified precision failed - model: {}, precision: {}, message: {}.'.format(
                    self._name, precision, str(e)
                )
            )
            return False

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
        curr_step = 0
        while True:
            for idx, sample in enumerate(self._dataloader):
                start = time.time()
                sample = sample.to(dtype=getattr(torch, precision.value))
                if self._gpu_available:
                    sample = sample.cuda()
                self._optimizer.zero_grad()
                output = self._model(sample)
                loss = self._loss_fn(output, self._target)
                loss.backward()
                self._optimizer.step()
                end = time.time()
                curr_step += 1
                if curr_step > self._args.num_warmup:
                    # Save the step time of every training/inference step, unit is millisecond.
                    duration.append((end - start) * 1000)
                if self._is_finished(curr_step, end):
                    return duration

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
        with torch.no_grad():
            self._model.eval()
            while True:
                for idx, sample in enumerate(self._dataloader):
                    start = time.time()
                    sample = getattr(sample, precision.value)
                    if self._gpu_available:
                        sample = sample.cuda()
                    self._model(sample)
                    if self._gpu_available:
                        torch.cuda.synchronize()
                    end = time.time()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every training/inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                    if self._is_finished(curr_step, end):
                        return duration


# Register CNN benchmarks.
# Reference: https://pytorch.org/vision/stable/models.html
for model in ['resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
    BenchmarkRegistry.register_benchmark('pytorch-' + model, PytorchCNN, parameters='--model_type ' + model)
