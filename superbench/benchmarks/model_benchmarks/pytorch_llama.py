#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch Llama2 model."""

import os
import random
import torch
from transformers import LlamaModel, LlamaConfig
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
except ImportError:
    te = None

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


class LlamaBenchmarkModel(torch.nn.Module):
    """The Llama model for benchmarking."""
    def __init__(self, config, num_classes):
        """Constructor.

        Args:
            config (LlamaConfig): Configurations of Llama model.
            num_classes (int): The number of objects for classification.
        """
        super().__init__()
        self._llama = LlamaModel(config)
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
        outputs = self._llama(input)
        result = self._linear(outputs[0])
        return result


class PytorchLlama(PytorchBase):
    """The Llama benchmark class."""
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

    def _enable_deterministic_training(self):
        """Enable deterministic training settings for reproducible results."""
        if hasattr(self._args, 'random_seed'):
            torch.manual_seed(self._args.random_seed)
            random.seed(self._args.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self._args.random_seed)
                torch.cuda.manual_seed_all(self._args.random_seed)

        # Enable deterministic algorithms
        # If SB_STRICT_DETERMINISM=1, raise on non-deterministic ops (required for cuBLAS/FlashAttention strictness)
        strict = os.environ.get('SB_STRICT_DETERMINISM', '0') == '1'
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def add_parser_arguments(self):
        """Add the Llama-specified arguments.

        Llama2 model reference: https://huggingface.co/docs/transformers/model_doc/llama2
        """
        super().add_parser_arguments()

        self._parser.add_argument('--num_classes', type=int, default=100, required=False, help='Num of class.')
        self._parser.add_argument('--hidden_size', type=int, default=1280, required=False, help='Hidden size.')
        self._parser.add_argument(
            '--num_hidden_layers', type=int, default=36, required=False, help='The number of hidden layers.'
        )
        self._parser.add_argument(
            '--num_attention_heads', type=int, default=20, required=False, help='The number of attention heads.'
        )
        self._parser.add_argument(
            '--intermediate_size',
            type=int,
            default=11008,
            required=False,
            help='Dimension of the MLP representations.'
        )
        self._parser.add_argument('--seq_len', type=int, default=512, required=False, help='Sequence length.')
        self._parser.add_argument(
            '--num_key_value_heads',
            type=int,
            default=None,
            required=False,
            help='The number of key_value heads that should be used to implement Grouped Query Attention.'
        )
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
        # Set seed before dataset generation if deterministic training is enabled
        if self._args.deterministic and hasattr(self._args, 'random_seed'):
            torch.manual_seed(self._args.random_seed)

        self._dataset = TorchRandomDataset(
            [self._args.sample_count, self._args.seq_len], self._world_size, dtype=torch.long
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
        # Enable deterministic training if requested
        if self._args.deterministic:
            self._enable_deterministic_training()

        self._config = LlamaConfig(
            hidden_size=self._args.hidden_size,
            num_hidden_layers=self._args.num_hidden_layers,
            num_attention_heads=self._args.num_attention_heads,
            num_key_value_heads=self._args.num_key_value_heads,
            intermediate_size=self._args.intermediate_size,
            max_position_embeddings=4096,    # Maximum sequence length that llama2 supports
            rms_norm_eps=1e-05,    # Llama2 default for epsilon used by the rms normalization layers
        )

        enable_fp8 = precision.name.startswith('FP8_')
        if enable_fp8 and te is None:
            logger.error(
                f'Create model with fp8 failed - model: {self._name}, precision: {precision},'
                ' message: Cannot find transformer_engine.'
            )
            return False
        if enable_fp8 and not self._gpu_available:
            logger.error(
                f'Create model with fp8 failed - model: {self._name}, precision: {precision},'
                ' message: FP8 is only supported on GPU.'
            )
            return False

        try:
            self._model = LlamaBenchmarkModel(self._config, self._args.num_classes)
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
        except BaseException as e:
            logger.error(
                'Create model with specified precision failed - model: {}, precision: {}, message: {}.'.format(
                    self._name, precision, str(e)
                )
            )
            return False

        # Generate targets - use seed if deterministic training is enabled
        if self._args.deterministic and hasattr(self._args, 'random_seed'):
            torch.manual_seed(self._args.random_seed + 1)  # +1 to avoid same seed as dataset

        self._target = torch.LongTensor(self._args.batch_size).random_(self._args.num_classes)
        if self._gpu_available:
            self._target = self._target.cuda()

        return True

    def _train_step(self, precision):
        """Define the training process.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Return:
            A tuple of (step_times_ms, info) where info may include per-step loss.
        """
        duration = []
        losses = []
        # Use a periodic cadence for any extra work (aligns with base default)
        check_frequency = 100
        curr_step = 0
        while True:
            for idx, sample in enumerate(self._dataloader):
                start = self._timer()
                if self._gpu_available:
                    sample = sample.cuda()
                self._optimizer.zero_grad()
                if self._fp8_recipe is not None:
                    with te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe):
                        output = self._model(sample)
                else:
                    output = self._model(sample)
                # Compute loss in float32 to avoid fp16 overflow/NaNs while keeping model in desired precision
                logits = output[range(self._args.batch_size), -1]
                loss = self._loss_fn(logits.float(), self._target)
                loss.backward()
                self._optimizer.step()
                end = self._timer()
                curr_step += 1
                if curr_step > self._args.num_warmup:
                    # Save the step time of every training step, unit is millisecond.
                    duration.append((end - start) * 1000)
                    # Record per-step loss for determinism checks
                    try:
                        losses.append(float(loss.detach().item()))
                    except Exception:
                        pass
                    # Simple periodic checksum when deterministic is enabled; log only.
                    if getattr(self._args, 'deterministic', False) and (curr_step % check_frequency == 0):
                        try:
                            checksum = sum(p.detach().float().sum().item() for p in self._model.parameters())
                            logger.info(f"Checksum at step {curr_step}: {checksum}")
                        except Exception:
                            # Never fail training due to checksum computation/logging
                            pass
                    self._log_step_time(curr_step, precision, duration)
                if self._is_finished(curr_step, end, check_frequency):
                    # Return optional info for additional raw metrics (loss)
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
                    start = self._timer()
                    if self._gpu_available:
                        sample = sample.cuda()
                    if self._fp8_recipe is not None:
                        with te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe):
                            self._model(sample)
                    else:
                        self._model(sample)
                    end = self._timer()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                        self._log_step_time(curr_step, precision, duration)
                    if self._is_finished(curr_step, end, check_frequency):
                        return duration

    def _process_info(self, model_action, precision, info):
        """Persist extra step-level signals (e.g., loss) into raw_data.

        Purpose:
            The base runner captures timing/throughput by default. When a step implementation
            returns additional information (like per-step loss), this hook translates that info
            into standardized raw_data entries (for example, fp16_train_loss) so tests and
            diagnostics can assert/inspect them consistently without altering summarization logic.

        Args:
            model_action: 'train' or 'inference'. Used to compose metric names.
            precision: model precision enum used to prefix metric names (e.g., fp16).
            info (dict): auxiliary data returned by _train_step/_inference_step, such as {'loss': [...]}.
        """
        try:
            if not info:
                return
            # Map precision enum to metric prefix
            precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'float64': 'fp64', 'bfloat16': 'bf16'}
            prec_value = precision.value if hasattr(precision, 'value') else str(precision)
            prefix = precision_metric.get(prec_value, prec_value)
            # Enum string formatting in base uses the enum directly; mimic that here
            metric_loss = f"{prefix}_{model_action}_loss"
            if 'loss' in info and isinstance(info['loss'], list) and len(info['loss']) > 0:
                # Store loss as raw data for assertions; do not add to summary statistics
                self._result.add_raw_data(metric_loss, info['loss'], self._args.log_raw_data)
        except Exception:
            # Be conservative: don't fail benchmark due to aux metrics
            pass


# Register Llama2 benchmark with 7b parameters.
BenchmarkRegistry.register_benchmark(
    'pytorch-llama2-7b',
    PytorchLlama,
    parameters='--hidden_size=4096 --num_hidden_layers=32 --num_attention_heads=32 --num_key_value_heads=32 \
        --intermediate_size=11008'
)

# Register Llama2 benchmark with 13b parameters.
BenchmarkRegistry.register_benchmark(
    'pytorch-llama2-13b',
    PytorchLlama,
    parameters='--hidden_size=5120 --num_hidden_layers=40 --num_attention_heads=40 --num_key_value_heads=40 \
        --intermediate_size=13824'
)

# Register Llama2 benchmark with 70b parameters.
BenchmarkRegistry.register_benchmark(
    'pytorch-llama2-70b',
    PytorchLlama,
    parameters='--hidden_size=8192 --num_hidden_layers=80 --num_attention_heads=64 --num_key_value_heads=8 \
        --intermediate_size=28672'
)
