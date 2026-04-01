# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch LSTM model."""

import torch

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


class LSTMBenchmarkModel(torch.nn.Module):
    """The LSTM model for benchmarking."""
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, num_classes):
        """Constructor.

        Args:
            input_size (int): The number of expected features in the input.
            hidden_size (int):  The number of features in the hidden state.
            num_layers  (int): The number of recurrent layers.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
            num_classes (int): The number of objects for classification.
        """
        super().__init__()
        self._lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self._linear = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, input):
        """Forward propagation function.

        Args:
            input (torch.FloatTensor): Tensor containing the features of the input sequence,
              shape (batch_size, sequence_length, input_size).

        Return:
            result (torch.FloatTensor): The output features from the last layer of the LSTM
              further processed by a Linear layer, shape (batch_size, num_classes).
        """
        self._lstm.flatten_parameters()
        outputs = self._lstm(input)
        result = self._linear(outputs[0][:, -1, :])
        return result


class PytorchLSTM(PytorchBase):
    """The LSTM benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._config = None
        self._supported_precision = [Precision.FLOAT32, Precision.FLOAT16]
        self._optimizer_type = Optimizer.SGD
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def add_parser_arguments(self):
        """Add the LSTM-specified arguments.

        LSTM model reference: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """
        super().add_parser_arguments()

        self._parser.add_argument(
            '--num_classes', type=int, default=100, required=False, help='The number of objects for classification.'
        )
        self._parser.add_argument(
            '--input_size', type=int, default=256, required=False, help='The number of expected features in the input.'
        )
        self._parser.add_argument(
            '--hidden_size', type=int, default=1024, required=False, help='The number of features in the hidden state.'
        )
        self._parser.add_argument(
            '--num_layers', type=int, default=8, required=False, help='The number of recurrent layers.'
        )

        self._parser.add_argument('--bidirectional', action='store_true', default=False, help='Bidirectional LSTM.')
        self._parser.add_argument('--seq_len', type=int, default=512, required=False, help='Sequence length.')

    def _generate_dataset(self):
        """Generate dataset for benchmarking according to shape info.

        Return:
            True if dataset is created successfully.
        """
        self._dataset = TorchRandomDataset(
            [self._args.sample_count, self._args.seq_len, self._args.input_size], self._world_size, dtype=torch.float32
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
        # Check if we should use HuggingFace model loading
        model_config = self._create_model_source_config(precision)
        if model_config and model_config.is_huggingface():
            return self._create_huggingface_model(model_config, precision)

        # Default in-house model creation
        return self._create_inhouse_model(precision)

    def _create_model_wrapper(self, hf_model, hf_config):
        """Create LSTM-specific model wrapper.

        Args:
            hf_model: The loaded HuggingFace LSTM model.
            hf_config: The HuggingFace model configuration.

        Returns:
            torch.nn.Module: Wrapped LSTM model with classification head.
        """
        class HFLSTMWrapper(torch.nn.Module):
            def __init__(self, lstm_model, vocab_size):
                super().__init__()
                self._lstm = lstm_model
                self._vocab_size = vocab_size

            def forward(self, input):
                # HF language models expect integer token IDs, not floats.
                # Convert float input to integer indices within vocab range.
                input_ids = input[:, :, 0].abs().mul(self._vocab_size - 1).long().clamp(0, self._vocab_size - 1)
                outputs = self._lstm(input_ids)
                # CausalLM models already include the output projection (logits).
                if hasattr(outputs, 'logits'):
                    return outputs.logits[:, -1, :]
                elif isinstance(outputs, tuple):
                    return outputs[0][:, -1, :]
                return outputs[:, -1, :]

        vocab_size = getattr(hf_config, 'vocab_size', 30522)
        # Use vocab_size as num_classes since CausalLM outputs logits over vocab
        self._args.num_classes = vocab_size
        return HFLSTMWrapper(hf_model, vocab_size)

    def _create_inhouse_model(self, precision):
        """Create in-house model (original implementation).

        Args:
            precision (Precision): precision of model and input data.

        Returns:
            bool: True if model created successfully, False otherwise.
        """
        try:
            self._model = LSTMBenchmarkModel(
                self._args.input_size, self._args.hidden_size, self._args.num_layers, self._args.bidirectional,
                self._args.num_classes
            )
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
        check_frequency = 100
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
                loss = self._loss_fn(output, self._target)
                loss.backward()
                self._optimizer.step()
                end = self._timer()
                curr_step += 1
                if curr_step > self._args.num_warmup:
                    # Save the step time of every training/inference step, unit is millisecond.
                    duration.append((end - start) * 1000)
                    self._log_step_time(curr_step, precision, duration)
                if self._is_finished(curr_step, end, check_frequency):
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
                    if self._is_finished(curr_step, end):
                        return duration


# Register LSTM benchmark.
BenchmarkRegistry.register_benchmark(
    'pytorch-lstm', PytorchLSTM, parameters='--input_size=256 --hidden_size=1024 --num_layers=8'
)
