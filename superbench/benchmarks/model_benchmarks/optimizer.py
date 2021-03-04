# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module to define optimizer type and to create optimizer instance."""

import torch

from superbench.benchmarks.context import Enum


class Optimizer(Enum):
    """The Enum class representing different optimizers."""
    SGD = 'sgd'
    ADAM = 'adam'
    ADAMW = 'adamw'


def create_torch_optimizer(optimizer, parameters):
    """Create optimizer instance according to optimizer type.

    Args:
        optimizer (Optimizer): Optimizer type.
        parameters (List[torch.Tensor]): List of torch.Tensor, get from model.parameters().
    """
    if optimizer == Optimizer.SGD:
        return torch.optim.SGD(parameters, lr=1e-5, momentum=0.9, weight_decay=1e-4, nesterov=True)
    elif optimizer == Optimizer.ADAM:
        return torch.optim.Adam(parameters, lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
    elif optimizer == Optimizer.ADAMW:
        return torch.optim.AdamW(parameters, lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
    else:
        return None
