# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Define the distributed modes."""

from enum import Enum


class DistributedMode(Enum):
    """The Enum class representing different distributed mode."""
    DDP = 'pytorch-ddp'
    HOROVOD = 'horovod'
