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


class BenchmarkType(Enum):
    """The Enum class representing different types of benchmarks."""
    MODEL = 'model'
    MICRO = 'micro'
    DOCKER = 'docker'


class Precision(Enum):
    """The Enum class representing different data precisions."""
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    BFLOAT16 = 'bfloat16'
    UINT8 = 'uint8'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'


class ModelAction(Enum):
    """The Enum class representing different model action."""
    TRAIN = 'train'
    INFEENCE = 'inference'


class BenchmarkContext():
    """Context class of all benchmarks.

    Containing all information to launch one benchmark.
    """
    def __init__(self, name, platform, parameters='', framework=Framework.NONE):
        """Constructor.

        Args:
            name (str): name of benchmark in config file.
            platform (Platform): Platform types like CUDA, ROCM.
            parameters (str): predefined parameters of benchmark.
            framework (Framework): Framework types like ONNX, PYTORCH.
        """
        self.__name = name
        self.__platform = platform
        self.__parameters = parameters
        self.__framework = framework

    @property
    def name(self):
        """Decoration function to access __name."""
        return self.__name

    @property
    def platform(self):
        """Decoration function to access __platform."""
        return self.__platform

    @property
    def parameters(self):
        """Decoration function to access __parameters."""
        return self.__parameters

    @property
    def framework(self):
        """Decoration function to access __framework."""
        return self.__framework
