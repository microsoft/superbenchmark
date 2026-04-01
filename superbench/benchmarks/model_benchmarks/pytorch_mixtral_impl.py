# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch Mixtral model implementation."""

import torch
from transformers import MixtralModel, MixtralConfig
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
except ImportError:
    te = None

from superbench.common.utils import logger
from superbench.benchmarks import Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


class MixtralBenchmarkModel(torch.nn.Module):
    """The Mixtral model for benchmarking."""
    def __init__(self, config, num_classes):
        """Constructor.

        Args:
            config (MixtralConfig): Configurations of Mixtral model.
            num_classes (int): The number of objects for classification.
        """
        super().__init__()
        self._Mixtral = MixtralModel(config)
        self._linear = torch.nn.Linear(config.hidden_size, num_classes)

    def forward(self, input):
        """Forward propagation function.

        Args:
            input (torch.LongTensor): Indices of input sequence tokens in the vocabulary,
              shape (batch_size, sequence_length).

        Return:
            result (torch.FloatTensor): Last layer hidden-state of the first token of the sequence
              (classification token) further processed by a Linear layer, shape (batch_size, hidden_size).
        """
        outputs = self._Mixtral(input)
        result = self._linear(outputs[0])
        return result


class PytorchMixtral(PytorchBase):
    """The Mixtral benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._config = None
        self._fp8_recipe = None
        self._supported_precision = [
            Precision.FLOAT32,
            Precision.FLOAT16,
            Precision.FP8_HYBRID,
            Precision.FP8_E4M3,
        ]
        self._optimizer_type = Optimizer.ADAMW
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def add_parser_arguments(self):
        """Add the Mixtral-specified arguments.

        Mixtral model reference: https://huggingface.co/docs/transformers/model_doc/Mixtral
        """
        super().add_parser_arguments()

        self._parser.add_argument('--num_classes', type=int, default=100, required=False, help='Num of class.')
        self._parser.add_argument('--hidden_size', type=int, default=4096, required=False, help='Hidden size.')
        self._parser.add_argument(
            '--num_hidden_layers', type=int, default=32, required=False, help='The number of hidden layers.'
        )
        self._parser.add_argument(
            '--num_attention_heads', type=int, default=32, required=False, help='The number of attention heads.'
        )
        self._parser.add_argument(
            '--intermediate_size',
            type=int,
            default=14336,
            required=False,
            help='Dimension of the MLP representations.'
        )
        self._parser.add_argument('--seq_len', type=int, default=512, required=False, help='Sequence length.')
        self._parser.add_argument(
            '--num_key_value_heads',
            type=int,
            default=8,
            required=False,
            help='The number of key_value heads that should be used to implement Grouped Query Attention.'
        )
        self._parser.add_argument(
            '--max_position_embeddings',
            type=int,
            default=None,
            required=False,
            help='Maximum sequence length that Mixtral supports'
        )
        self._parser.add_argument(
            '--router_aux_loss_coef',
            type=float,
            default=0.001,
            required=False,
            help='The aux loss factor for the total loss.'
        )

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
        self._dataset = TorchRandomDataset(
            [self._args.sample_count, self._args.seq_len], self._world_size, dtype=torch.long
        )
        if len(self._dataset) == 0:
            logger.error('Generate random dataset failed - model: {}'.format(self._name))
            return False

        return True

    def _customize_hf_config(self, hf_config):
        """Override num_hidden_layers if a non-default value was specified."""
        if self._args.num_hidden_layers != 32:
            hf_config.num_hidden_layers = self._args.num_hidden_layers
        return hf_config

    def _create_model_wrapper(self, hf_model, hf_config):
        """Create Mixtral-specific model wrapper for HuggingFace models.

        Args:
            hf_model: The loaded HuggingFace Mixtral model.
            hf_config: The HuggingFace model configuration.

        Returns:
            torch.nn.Module: Wrapped model with classification head.
        """
        hidden_size = hf_config.hidden_size

        class HFMixtralWrapper(torch.nn.Module):
            def __init__(self, hf_model, num_classes, hidden_size):
                super().__init__()
                self.model = hf_model
                self.classifier = torch.nn.Linear(hidden_size, num_classes)

            def forward(self, input):
                outputs = self.model(input)
                return self.classifier(outputs[0])

        return HFMixtralWrapper(hf_model, self._args.num_classes, hidden_size)

    def _create_model(self, precision):
        """Construct the model for benchmarking.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.
        """
        self._config = self._build_config()

        # Check if using HuggingFace model source
        model_config = self._create_model_source_config(precision)
        if model_config and model_config.source == 'huggingface':
            return self._create_huggingface_model(model_config, precision)

        if not self._check_fp8_support(precision):
            return False

        try:
            self._model = self._instantiate_model()
            self._postprocess_model(precision)
        except Exception as e:
            logger.error(
                'Create model with specified precision failed - model: {}, precision: {}, message: {}.'.format(
                    self._name, precision, str(e)
                )
            )
            return False

        self._setup_target()
        return True

    def _build_config(self):
        return MixtralConfig(
            hidden_size=self._args.hidden_size,
            num_hidden_layers=self._args.num_hidden_layers,
            num_attention_heads=self._args.num_attention_heads,
            num_key_value_heads=self._args.num_key_value_heads,
            intermediate_size=self._args.intermediate_size,
            max_position_embeddings=self._args.max_position_embeddings,
            router_aux_loss_coef=self._args.router_aux_loss_coef,
        )

    def _check_fp8_support(self, precision):
        enable_fp8 = precision.name.startswith('FP8_')
        if enable_fp8 and te is None:
            logger.error(
                f'Create model with fp8 failed - model: {self._name}, precision: {precision}, '
                'message: Cannot find transformer_engine.'
            )
            return False
        if enable_fp8 and not self._gpu_available:
            logger.error(
                f'Create model with fp8 failed - model: {self._name}, precision: {precision}, '
                'message: FP8 is only supported on GPU.'
            )
            return False
        return True

    def _instantiate_model(self):
        return MixtralBenchmarkModel(self._config, self._args.num_classes)

    def _postprocess_model(self, precision):
        enable_fp8 = precision.name.startswith('FP8_')
        if enable_fp8:
            self._fp8_recipe = DelayedScaling(
                fp8_format=Format[precision.name.strip('FP8_')],
                amax_history_len=16,
                amax_compute_algo='max',
            )
            self._to_te_model(self._model.to(dtype=torch.float16))
        else:
            self._model = self._model.to(dtype=getattr(torch, precision.value))
        if self._gpu_available:
            self._model = self._model.cuda()

    def _setup_target(self):
        """Set up target tensor using the shared deterministic-aware helper."""
        self._target = self._create_target(self._args.num_classes)

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
                start = self._timer()
                if self._gpu_available:
                    sample = sample.cuda()
                if self._args.exclude_copy_time:
                    start = self._timer()
                self._optimizer.zero_grad()
                if self._fp8_recipe is not None:
                    with te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe):
                        output = self._model(sample)
                else:
                    output = self._model(sample)
                logits = output[range(self._args.batch_size), -1]
                # Use FP32 logits for loss only when determinism is enabled; otherwise
                # keep logits in their native precision to preserve benchmark semantics.
                enable_determinism = getattr(self._args, 'enable_determinism', False)
                logits_for_loss = logits.float() if enable_determinism else logits
                loss = self._loss_fn(logits_for_loss, self._target)
                loss.backward()
                self._optimizer.step()
                end = self._timer()
                curr_step += 1
                if curr_step > self._args.num_warmup:
                    duration.append((end - start) * 1000)
                    self.record_determinism_fingerprint(curr_step, loss, logits, periodic, self._args.check_frequency)
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
                    start = self._timer()
                    if self._gpu_available:
                        sample = sample.cuda()
                    if self._args.exclude_copy_time:
                        start = self._timer()
                    if self._fp8_recipe is not None:
                        with te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe):
                            self._model(sample)
                    else:
                        self._model(sample)
                    end = self._timer()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every training/inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                        self._log_step_time(curr_step, precision, duration)
                    if self._is_finished(curr_step, end, self._args.check_frequency):
                        return duration
