# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the generic PyTorch HuggingFace language-model benchmark.

This benchmark loads any decoder-style language model (e.g. GPT-2, GPT-J, GPT-NeoX,
Llama 2/3) directly from the HuggingFace Hub and benchmarks its train/inference
throughput the same way the other PyTorch model benchmarks do. The model is only
downloaded and run when the pre-flight memory check estimates it fits on the
available hardware, so large checkpoints are rejected before any weights are pulled.
"""

import torch

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


class PytorchHFLanguageModel(PytorchBase):
    """Benchmark class for GPT/Llama language models loaded from HuggingFace Hub."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._config = None
        self._supported_precision = [
            Precision.FLOAT32,
            Precision.FLOAT16,
            Precision.BFLOAT16,
        ]
        self._optimizer_type = Optimizer.ADAMW
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def add_parser_arguments(self):
        """Add the HuggingFace language-model-specific arguments."""
        super().add_parser_arguments()

        self._parser.add_argument('--num_classes', type=int, default=100, required=False, help='Num of class.')
        self._parser.add_argument('--seq_len', type=int, default=512, required=False, help='Sequence length.')

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

    def _create_model(self, precision):
        """Construct the model for benchmarking.

        This benchmark only supports HuggingFace-sourced models. A valid
        ``--model_identifier`` must be supplied together with
        ``--model_source huggingface``.

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.

        Returns:
            bool: True if model created successfully, False otherwise.
        """
        model_config = self._create_model_source_config(precision)
        if not (model_config and model_config.is_huggingface()):
            logger.error(
                'HuggingFace language-model benchmark requires "--model_source huggingface" and '
                '"--model_identifier <org/model-name>" - model: {}.'.format(self._name)
            )
            return False

        return self._create_huggingface_model(model_config, precision)

    def _create_model_wrapper(self, hf_model, hf_config):
        """Create a generic classification-head wrapper for the loaded HuggingFace model.

        Both GPT-style and Llama-style base models return the last hidden state as the
        first element of their output, so a single wrapper works across architectures.

        Args:
            hf_model: The loaded HuggingFace base model.
            hf_config: The HuggingFace model configuration.

        Returns:
            torch.nn.Module: Wrapped model with a linear classification head.
        """
        hidden_size = (
            getattr(hf_config, 'hidden_size', None) or getattr(hf_config, 'n_embd', None)
            or getattr(hf_config, 'd_model', None)
        )
        if hidden_size is None:
            raise ValueError(
                f'Could not determine hidden size from config for model {self._name}. '
                'Expected one of: hidden_size, n_embd, d_model.'
            )

        class HFLanguageModelWrapper(torch.nn.Module):
            """Wrapper adding a classification head on top of a HuggingFace base model."""
            def __init__(self, base_model, hidden_size, num_classes):
                super().__init__()
                self.model = base_model
                self.classifier = torch.nn.Linear(hidden_size, num_classes)

            def forward(self, input):
                outputs = self.model(input)
                return self.classifier(outputs[0])

        return HFLanguageModelWrapper(hf_model, hidden_size, self._args.num_classes)

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
                    # Save the step time of every training step, unit is millisecond.
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
                    self._model(sample)
                    end = self._timer()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                        self._log_step_time(curr_step, precision, duration)
                    if self._is_finished(curr_step, end, self._args.check_frequency):
                        return duration


# Generic entry — supply "--model_identifier <org/model-name>" (any GPT/Llama checkpoint)
# via the benchmark parameters (e.g. a SuperBench config file). "--model_source huggingface"
# is baked in so only the identifier is required.
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-lm',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface',
)

# GPT family (publicly available checkpoints).
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-gpt2',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier openai-community/gpt2',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-gpt2-medium',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier openai-community/gpt2-medium',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-gpt2-large',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier openai-community/gpt2-large',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-gpt2-xl',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier openai-community/gpt2-xl',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-gptj-6b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier EleutherAI/gpt-j-6b',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-gpt-neox-20b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier EleutherAI/gpt-neox-20b',
)

# Llama family (gated checkpoints — require HF_TOKEN to download).
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-llama2-7b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier meta-llama/Llama-2-7b-hf',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-llama2-13b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier meta-llama/Llama-2-13b-hf',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-llama2-70b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier meta-llama/Llama-2-70b-hf',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-llama3-8b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier meta-llama/Meta-Llama-3-8B',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-llama3.1-8b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier meta-llama/Llama-3.1-8B',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-llama3.2-1b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier meta-llama/Llama-3.2-1B',
)
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-llama3.2-3b',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier meta-llama/Llama-3.2-3B',
)
