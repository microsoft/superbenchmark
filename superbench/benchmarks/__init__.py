# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exposes interfaces of benchmarks used by SuperBench executor."""

from .return_code import ReturnCode
from .context import Platform, Framework, Precision, ModelAction, BenchmarkType, BenchmarkContext
from .registry import BenchmarkRegistry

__all__ = [
    'ReturnCode', 'Platform', 'Framework', 'BenchmarkType', 'Precision', 'ModelAction', 'BenchmarkContext',
    'BenchmarkRegistry'
]
