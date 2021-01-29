# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkResult module."""

from superbench.benchmarks import BenchmarkResult


def do_serialize_deserialize(result):
    """Do serialization/deserialization and compare the results.

    Args:
        result (BenchmarkResult): input result
    """
    result_se = result.to_string()
    result_de = BenchmarkResult.from_string(result_se)
    assert (result == result_de)


def test_benchmark_result_format():
    """Test the format of BenchmarkResult."""
    # Result with one metric.
    result = BenchmarkResult('pytorch-bert-base1')
    result.add_result('metric1', '300')
    do_serialize_deserialize(result)

    # Result with two metrics.
    result = BenchmarkResult('pytorch-bert-base2')
    result.add_result('metric1', '100')
    result.add_result('metric2', '200')
    do_serialize_deserialize(result)

    # Empty result.
    result_de = BenchmarkResult.from_string('{}')
    assert (result_de is None)
