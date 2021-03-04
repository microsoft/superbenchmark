# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for statuses of benchmarks."""

from superbench.benchmarks.context import Enum


class ReturnCode(Enum):
    """The Enum class representing benchmark status."""
    # Common return codes.
    SUCCESS = 0
    INVALID_ARGUMENT = 1
    INVALID_BENCHMARK_TYPE = 2
    INVALID_BENCHMARK_RESULT = 3
    # Return codes related with model benchmarks.
    NO_SUPPORTED_PRECISION = 10
    MODEL_TRAIN_FAILURE = 11
    MODEL_INFERENCE_FAILURE = 12
    DISTRIBUTED_SETTING_INIT_FAILURE = 13
    DATASET_GENERATION_FAILURE = 14
    DATALOADER_INIT_FAILURE = 15
    OPTIMIZER_CREATION_FAILURE = 16
    MODEL_CREATION_FAILURE = 17
