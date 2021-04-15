# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A module containing all the e2e model related benchmarks."""

from superbench.benchmarks.model_benchmarks.model_base import ModelBenchmark
from superbench.benchmarks.model_benchmarks.pytorch_bert import PytorchBERT
from superbench.benchmarks.model_benchmarks.pytorch_cnn import PytorchCNN

__all__ = ['ModelBenchmark', 'PytorchBERT', 'PytorchCNN']
