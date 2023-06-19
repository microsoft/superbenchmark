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

    def __str__(self):
        """Value as the string."""
        return str(self.value)


class Platform(Enum):
    """The Enum class representing different platforms."""
    CPU = 'CPU'
    CUDA = 'CUDA'
    ROCM = 'ROCm'
    DIRECTX = 'DirectX'


class Framework(Enum):
    """The Enum class representing different frameworks."""
    ONNXRUNTIME = 'onnxruntime'
    PYTORCH = 'pytorch'
    TENSORFLOW1 = 'tf1'
    TENSORFLOW2 = 'tf2'
    NONE = 'none'


class BenchmarkType(Enum):
    """The Enum class representing different types of benchmarks."""
    MODEL = 'model'
    MICRO = 'micro'
    DOCKER = 'docker'


class Precision(Enum):
    """The Enum class representing different data precisions."""
    FP8_HYBRID = 'fp8_hybrid'
    FP8_E4M3 = 'fp8_e4m3'
    FP8_E5M2 = 'fp8_e5m2'
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
    """The Enum class representing different model process."""
    TRAIN = 'train'
    INFERENCE = 'inference'


class DistributedImpl(Enum):
    """The Enum class representing different distributed implementations."""
    DDP = 'ddp'
    MIRRORED = 'mirrored'
    MW_MIRRORED = 'multiworkermirrored'
    PS = 'parameterserver'
    HOROVOD = 'horovod'


class DistributedBackend(Enum):
    """The Enum class representing different distributed backends."""
    NCCL = 'nccl'
    MPI = 'mpi'
    GLOO = 'gloo'


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
            framework (Framework): Framework types like ONNXRUNTIME, PYTORCH.
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
