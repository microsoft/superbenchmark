# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch BERT model."""

import time

import torch
from transformers import BertModel, BertConfig

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_base import PytorchBase
from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


class BertBenchmarkModel(torch.nn.Module):
    """The BERT model for benchmarking."""
    def __init__(self, config, num_class):
        """Constructor.

        Args:
            config (BertConfig): Configurations of BERT model.
            num_class (int): The number of objects for classification.
        """
        super().__init__()
        self._bert = BertModel(config)
        self._linear = torch.nn.Linear(config.hidden_size, num_class)

    def forward(self, input):
        """Forward propagation function.

        Args:
            input (torch.LongTensor): Indices of input sequence tokens in the vocabulary,
              shape (batch_size, sequence_length).

        Return:
            result (torch.FloatTensor): Last layer hidden-state of the first token of the sequence
              (classification token) further processed by a Linear layer, shape (batch_size, hidden_size).
        """
        outputs = self._bert(input)
        result = self._linear(outputs[1])
        return result


class PytorchBERT(PytorchBase):
    """The BERT benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._config = None
        self._supported_precision = [Precision.FLOAT32, Precision.FLOAT16]
        self._optimizer_type = Optimizer.ADAMW
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def add_parser_arguments(self):
        """Add the BERT-specified arguments.

        BERT model reference: https://huggingface.co/transformers/model_doc/bert.html
        """
        super().add_parser_arguments()

        self._parser.add_argument('--num_classes', type=int, default=100, required=False, help='Num of class.')
        self._parser.add_argument('--hidden_size', type=int, default=1024, required=False, help='Hidden size.')
        self._parser.add_argument(
            '--num_hidden_layers', type=int, default=24, required=False, help='The number of hidden layers.'
        )
        self._parser.add_argument(
            '--num_attention_heads', type=int, default=16, required=False, help='The number of attention heads.'
        )
        self._parser.add_argument(
            '--intermediate_size', type=int, default=4096, required=False, help='Intermediate size.'
        )
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

        Args:
            precision (Precision): precision of model and input data, such as float32, float16.
        """
        self._config = BertConfig(
            hidden_size=self._args.hidden_size,
            num_hidden_layers=self._args.num_hidden_layers,
            num_attention_heads=self._args.num_attention_heads,
            intermediate_size=self._args.intermediate_size
        )

        try:
            self._model = BertBenchmarkModel(self._config, self._args.num_classes)
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
                    torch.cuda.synchronize()
                    start = time.time()
                    sample = sample.cuda()
                    self._model(sample)
                    torch.cuda.synchronize()
                    end = time.time()
                    curr_step += 1
                    if curr_step > self._args.num_warmup:
                        # Save the step time of every training/inference step, unit is millisecond.
                        duration.append((end - start) * 1000)
                    if self._is_finished(curr_step, end):
                        return duration


# Register BERT Large benchmark.
# Reference: https://huggingface.co/transformers/pretrained_models.html
BenchmarkRegistry.register_benchmark(
    'pytorch-bert-large',
    PytorchBERT,
    parameters='--hidden_size=1024 --num_hidden_layers=24 --num_attention_heads=16 --intermediate_size=4096'
)

# Register BERT Base benchmark.
# Reference: https://huggingface.co/transformers/pretrained_models.html
BenchmarkRegistry.register_benchmark(
    'pytorch-bert-base',
    PytorchBERT,
    parameters='--hidden_size=768 --num_hidden_layers=12 --num_attention_heads=12 --intermediate_size=3072'
)
