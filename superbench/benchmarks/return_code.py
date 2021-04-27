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
    DISTRIBUTED_SETTING_DESTROY_FAILURE = 14
    DATASET_GENERATION_FAILURE = 15
    DATALOADER_INIT_FAILURE = 16
    OPTIMIZER_CREATION_FAILURE = 17
    MODEL_CREATION_FAILURE = 18
    # Return codes related with micro benchmarks.
    MICROBENCHMARK_BINARY_NOT_EXIST = 30
    MICROBENCHMARK_EXECUTION_FAILURE = 31
    MICROBENCHMARK_RESULT_PARSING_FAILURE = 32
