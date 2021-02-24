# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for unified context of benchmarks."""

import enum


class Enum(enum.Enum):
    """Customized Enum class."""
    @classmethod
    def get_values(cls):
        """Return the value list."""
        values = [item.value for item in cls]
        return values


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
