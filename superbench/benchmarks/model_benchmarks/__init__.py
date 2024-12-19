# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A module containing all the e2e model related benchmarks."""

import sys

from superbench.benchmarks.model_benchmarks.model_base import ModelBenchmark
from superbench.benchmarks.model_benchmarks.pytorch_bert import PytorchBERT
from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import PytorchGPT2
from superbench.benchmarks.model_benchmarks.pytorch_cnn import PytorchCNN
from superbench.benchmarks.model_benchmarks.pytorch_lstm import PytorchLSTM
from superbench.benchmarks.model_benchmarks.megatron_gpt3 import MegatronGPT
from superbench.benchmarks.model_benchmarks.pytorch_llama import PytorchLlama

__all__ = [
    'ModelBenchmark', 'PytorchBERT', 'PytorchGPT2', 'PytorchCNN', 'PytorchLSTM', 'MegatronGPT', 'PytorchLlama',
    'PytorchMixtral'
]

# Check for Python version > 3.7 and conditionally import PytorchMixtral
if sys.version_info > (3, 7):
    from superbench.benchmarks.model_benchmarks.pytorch_mixtral import PytorchMixtral
    __all__.append('PytorchMixtral')
