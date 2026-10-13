# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Precision coverage for Transformer Engine consumers."""

import pytest
import torch

from superbench.benchmarks import Precision
from superbench.benchmarks.model_benchmarks.model_base import Optimizer
from superbench.benchmarks.model_benchmarks.pytorch_bert import PytorchBERT
from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import PytorchGPT2
from superbench.benchmarks.model_benchmarks.pytorch_llama import PytorchLlama


@pytest.mark.parametrize('benchmark', [PytorchBERT, PytorchGPT2, PytorchLlama])
def test_transformer_bfloat16_supported(benchmark):
    """BF16 must not be silently skipped by the generic precision filter."""
    instance = benchmark('precision-test')
    assert Precision.BFLOAT16 in instance._supported_precision
    instance.add_parser_arguments()
    valid, arguments, _ = instance.parse_args()
    assert valid
    assert Precision.BFLOAT16 not in arguments.precision


@pytest.mark.parametrize('optimizer', [Optimizer.ADAM, Optimizer.ADAMW])
@pytest.mark.parametrize('parameters,epsilon', [('', 1e-8), ('--optimizer_eps 1e-4', 1e-4)])
def test_optimizer_epsilon(optimizer, parameters, epsilon):
    """Preserve the default and permit an explicitly FP16-representable epsilon."""
    benchmark = PytorchGPT2('precision-test', parameters)
    benchmark.add_parser_arguments()
    valid, benchmark._args, _ = benchmark.parse_args()
    assert valid
    benchmark._model = torch.nn.Linear(16, 16)
    benchmark._optimizer_type = optimizer
    assert benchmark._create_optimizer()
    assert benchmark._optimizer.param_groups[0]['eps'] == epsilon
