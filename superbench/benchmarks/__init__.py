# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exposes interfaces of benchmarks used by SuperBench executor."""

from superbench.benchmarks.return_code import ReturnCode
from superbench.benchmarks.context import Platform, Framework, Precision, ModelAction, BenchmarkType, BenchmarkContext
from superbench.benchmarks.registry import BenchmarkRegistry
import superbench.benchmarks.model_benchmarks    # noqa pylint: disable=unused-import
import superbench.benchmarks.micro_benchmarks    # noqa pylint: disable=unused-import
import superbench.benchmarks.docker_benchmarks    # noqa pylint: disable=unused-import

__all__ = [
    'ReturnCode', 'Platform', 'Framework', 'BenchmarkType', 'Precision', 'ModelAction', 'BenchmarkContext',
    'BenchmarkRegistry'
]
