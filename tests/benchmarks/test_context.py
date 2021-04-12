# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkResult module."""

from superbench.benchmarks import BenchmarkContext, Platform, Framework


def test_benchmark_context():
    """Test BenchmarkContext class."""
    context = BenchmarkContext('pytorch-bert-large', Platform.CUDA, '--batch_size 8', framework=Framework.PYTORCH)
    assert (context.name == 'pytorch-bert-large')
    assert (context.platform == Platform.CUDA)
    assert (context.parameters == '--batch_size 8')
    assert (context.framework == Framework.PYTORCH)
