# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for unified context of benchmarks."""

from enum import Enum


class Platform(Enum):
    """The Enum class representing different platforms."""
    CPU = 'CPU'
    CUDA = 'CUDA'
    ROCM = 'ROCm'


class Framework(Enum):
    """The Enum class representing different frameworks."""
    ONNX = 'onnx'
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tf'
    NONE = 'none'


class BenchmarkContext():
    """Context class of all benchmarks.

    Containing all information to launch one benchmark.
    """
    def __init__(self,
                 name,
                 platform,
                 parameters='',
                 framework=Framework.NONE):
        """Constructor.

        Args:
            name (str): name of benchmark in config file.
            platform (Platform): Platform types like CUDA, ROCM.
            parameters (str): predefined parameters of benchmark.
            framework (Framework): Framework types like ONNX, PYTORCH.
        """
        self.name = name
        self.platform = platform
        self.parameters = parameters
        self.framework = framework
