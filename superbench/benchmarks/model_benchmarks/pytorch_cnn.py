# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch CNN models."""

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

    def add_parser_arguments(self):
        """Add the CNN-specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument('--model_type', type=str, required=True, help='The cnn benchmark to run.')
        self._parser.add_argument('--image_size', type=int, default=224, required=False, help='Image size.')
        self._parser.add_argument('--num_classes', type=int, default=1000, required=False, help='Num of class.')

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
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

        Returns:
            bool: True if model is created successfully.
        """
        # Check if using HuggingFace model source
        model_config = self._create_model_source_config(precision)
        if model_config and model_config.source == 'huggingface':
            return self._create_huggingface_model(model_config, precision)
        else:
            return self._create_inhouse_model(precision)

    def _create_model_wrapper(self, hf_model, hf_config):
        """Create CNN-specific model wrapper (ResNet/DenseNet).

        Args:
            hf_model: The loaded HuggingFace CNN model.
            hf_config: The HuggingFace model configuration.

        Returns:
            torch.nn.Module: Wrapped CNN model with classification head.
        """
        # Get the feature dimension from the model config
        if hasattr(hf_config, 'num_features'):
            feature_dim = hf_config.num_features
        elif hasattr(hf_config, 'hidden_sizes') and hf_config.hidden_sizes:
            feature_dim = hf_config.hidden_sizes[-1]
        else:
            # Fallback: inspect the actual model to get pooler output dimension
            model_dtype = next(hf_model.parameters()).dtype
            dummy_input = torch.randn(1, 3, 224, 224, dtype=model_dtype, device=next(hf_model.parameters()).device)
            with torch.no_grad():
                dummy_output = hf_model(dummy_input)
                feature_dim = dummy_output.pooler_output.shape[1]
                logger.info(f'Detected feature dimension from model output: {feature_dim}')

        class HFResNetWrapper(torch.nn.Module):
            """Wrapper for HuggingFace ResNet/DenseNet model."""
            def __init__(self, hf_model, num_classes, feature_dim):
                super().__init__()
                self.model = hf_model
                self.classifier = torch.nn.Linear(feature_dim, num_classes)

            def forward(self, pixel_values):
                outputs = self.model(pixel_values)
                pooled = outputs.pooler_output.flatten(1)
                return self.classifier(pooled)

        return HFResNetWrapper(hf_model, self._args.num_classes, feature_dim)

    def _create_inhouse_model(self, precision):
        """Create in-house torchvision ResNet model.

        Args:
            precision (Precision): precision of model and input data.

        Returns:
            bool: True if model is created successfully.
        """
        try:
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

        self._target = self._create_target(self._args.num_classes)

        return True

    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            A tuple of (step_times_ms, info) of every training step.
        """
        duration = []
        periodic = {'loss': [], 'act_mean': [], 'step': []}
        curr_step = 0
        while True:
            for idx, sample in enumerate(self._dataloader):
                sample = sample.to(dtype=getattr(torch, precision.value))
                start = self._timer()
                if self._gpu_available:
                    sample = sample.cuda()
                if self._args.exclude_copy_time:
                    start = self._timer()
                self._optimizer.zero_grad()
                output = self._model(sample)
                enable_determinism = getattr(self._args, 'enable_determinism', False)
                logits_for_loss = output.float() if enable_determinism else output
                loss = self._loss_fn(logits_for_loss, self._target)
                loss.backward()
                self._optimizer.step()
                end = self._timer()
                curr_step += 1
                if curr_step > self._args.num_warmup:
                    # Save the step time of every training/inference step, unit is millisecond.
                    duration.append((end - start) * 1000)
                    self.record_determinism_fingerprint(curr_step, loss, output, periodic, self._args.check_frequency)
                    self._log_step_time(curr_step, precision, duration)
                if self._is_finished(curr_step, end, self._args.check_frequency):
                    return duration, self._finalize_periodic_logging(periodic)

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
                    sample = sample.to(dtype=getattr(torch, precision.value))
                    start = self._timer()
                    if self._gpu_available:
                        sample = sample.cuda()
                    if self._args.exclude_copy_time:
                        start = self._timer()
                    self._model(sample)
                    end = self._timer()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every training/inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                        self._log_step_time(curr_step, precision, duration)
                    if self._is_finished(curr_step, end, self._args.check_frequency):
                        return duration


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
