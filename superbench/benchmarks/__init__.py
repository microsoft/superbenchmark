# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exposes interfaces of benchmarks used by SuperBench executor."""

import importlib

from superbench.benchmarks.return_code import ReturnCode
from superbench.benchmarks.context import Platform, Framework, Precision, ModelAction, \
    DistributedImpl, DistributedBackend, BenchmarkType, BenchmarkContext
from superbench.benchmarks.reducer import ReduceType, Reducer
from superbench.common.utils import LazyImport

BenchmarkRegistry = LazyImport(
    'superbench.benchmarks.registry', 'BenchmarkRegistry', lambda: list(
        map(
            importlib.import_module, [
                'superbench.benchmarks.{}'.format(module)
                for module in ['model_benchmarks', 'micro_benchmarks', 'docker_benchmarks']
            ]
        )
    )
)

__all__ = [
    'ReturnCode', 'Platform', 'Framework', 'BenchmarkType', 'Precision', 'ModelAction', 'DistributedImpl',
    'DistributedBackend', 'BenchmarkContext', 'BenchmarkRegistry', 'ReduceType', 'Reducer'
]
