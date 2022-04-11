# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A module containing all the benchmarks packaged in docker."""

from superbench.benchmarks.docker_benchmarks.docker_base import DockerBenchmark, CudaDockerBenchmark, \
    RocmDockerBenchmark
from superbench.benchmarks.docker_benchmarks.rocm_onnxruntime_performance import RocmOnnxRuntimeModelBenchmark
from superbench.benchmarks.docker_benchmarks.fambench import FAMBenchBenchmark

__all__ = [
    'DockerBenchmark', 'CudaDockerBenchmark', 'RocmDockerBenchmark', 'RocmOnnxRuntimeModelBenchmark',
    'FAMBenchBenchmark'
]
