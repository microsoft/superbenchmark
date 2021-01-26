# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum


class Platform(Enum):
    '''The Enum class representing different platfroms.
    '''
    CPU = 'CPU'
    CUDA = 'CUDA'
    ROCM = 'ROCm'
    UNKNOWN = 'Unknown'


class Framework(Enum):
    '''The Enum class representing different frameworks.
    '''
    ONNX = 'onnx'
    PYTORCH = 'pytorch'
    TENSORFLOW = 'tf'
    NONE = 'none'


class BenchmarkContext():
    '''Result class of all benchmarks, defines the result format.

    Args:
        name: name of benchmark.
        return_code: return code of benchmark.
        run_count: run count of benchmark, all runs will be organized as array.
        start_time: starting time of benchmark.
        end_time: ending time of benchmark.
        raw_data: keys are metrics, values are arrays for N runs.
        result: keys are metrics, values are arrays for N runs.
    '''
    def __init__(self, name, platform, parameters='',
                 framework=Framework.NONE):
        self.name = name
        self.platform = platform
        self.parameters = parameters
        self.framework = framework
