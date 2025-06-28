# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch Mixtral model gate."""

import sys
from superbench.benchmarks import BenchmarkRegistry

if sys.version_info < (3, 8):
    MixtralBenchmarkModel = None
    PytorchMixtral = None
else:
    from superbench.benchmarks.model_benchmarks.pytorch_mixtral_impl import MixtralBenchmarkModel, PytorchMixtral

    # Register Mixtral benchmark with 8x7b parameters.
    # Ref: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
    BenchmarkRegistry.register_benchmark(
        'pytorch-mixtral-8x7b',
        PytorchMixtral,
        parameters='--hidden_size=4096 --num_hidden_layers=32 --num_attention_heads=32 --intermediate_size=14336 \
            --num_key_value_heads=8 --max_position_embeddings=32768 --router_aux_loss_coef=0.02'
    )

    # Register Mixtral benchmark with 8x22b parameters.
    # Ref: https://huggingface.co/mistralai/Mixtral-8x22B-v0.1/blob/main/config.json
    BenchmarkRegistry.register_benchmark(
        'pytorch-mixtral-8x22b',
        PytorchMixtral,
        parameters='--hidden_size=6144 --num_hidden_layers=56 --num_attention_heads=48 --intermediate_size=16384 \
            --num_key_value_heads=8 --max_position_embeddings=65536 --router_aux_loss_coef=0.001'
    )

__all__ = ['MixtralBenchmarkModel', 'PytorchMixtral']
