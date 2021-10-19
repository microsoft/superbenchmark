# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A module containing all the benchmarks packaged in docker."""

from superbench.benchmarks.docker_benchmarks.docker_base import CudaDockerBenchmark, RocmDockerBenchmark

__all__ = ['CudaDockerBenchmark', 'RocmDockerBenchmark']
