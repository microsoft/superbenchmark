# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkResult module."""

import os

from superbench.benchmarks import BenchmarkType, ReturnCode, ReduceType
from superbench.benchmarks.result import BenchmarkResult


def test_add_raw_data():
    """Test interface BenchmarkResult.add_raw_data()."""
    result = BenchmarkResult('micro', BenchmarkType.MICRO, ReturnCode.SUCCESS)
    result.add_raw_data('metric1', 'raw log 1', False)
    result.add_raw_data('metric1', 'raw log 2', False)
    assert (result.raw_data['metric1'][0] == 'raw log 1')
    assert (result.raw_data['metric1'][1] == 'raw log 2')
    assert (result.type == BenchmarkType.MICRO)
    assert (result.return_code == ReturnCode.SUCCESS)

    result = BenchmarkResult('model', BenchmarkType.MODEL, ReturnCode.SUCCESS)
    result.add_raw_data('metric1', [1, 2, 3], False)
    result.add_raw_data('metric1', [4, 5, 6], False)
    assert (result.raw_data['metric1'][0] == [1, 2, 3])
    assert (result.raw_data['metric1'][1] == [4, 5, 6])
    assert (result.type == BenchmarkType.MODEL)
    assert (result.return_code == ReturnCode.SUCCESS)

    # Test log_raw_data = True.
    result = BenchmarkResult('micro', BenchmarkType.MICRO, ReturnCode.SUCCESS)
    result.add_raw_data('metric1', 'raw log 1', True)
    result.add_raw_data('metric1', 'raw log 2', True)
    assert (result.type == BenchmarkType.MICRO)
    assert (result.return_code == ReturnCode.SUCCESS)
    raw_data_file = os.path.join(os.getcwd(), 'rawdata.log')
    assert (os.path.isfile(raw_data_file))
    os.remove(raw_data_file)


def test_add_result():
    """Test interface BenchmarkResult.add_result()."""
    result = BenchmarkResult('micro', BenchmarkType.MICRO, ReturnCode.SUCCESS)
    result.add_result('metric1', 300)
    result.add_result('metric1', 200)
    assert (result.result['metric1'][0] == 300)
    assert (result.result['metric1'][1] == 200)


def test_set_timestamp():
    """Test interface BenchmarkResult.set_timestamp()."""
    result = BenchmarkResult('micro', BenchmarkType.MICRO, ReturnCode.SUCCESS)
    start_time = '2021-02-03 16:59:49'
    end_time = '2021-02-03 17:00:08'
    result.set_timestamp(start_time, end_time)
    assert (result.start_time == start_time)
    assert (result.end_time == end_time)


def test_set_benchmark_type():
    """Test interface BenchmarkResult.set_benchmark_type()."""
    result = BenchmarkResult('micro', BenchmarkType.MICRO, ReturnCode.SUCCESS)
    result.set_benchmark_type(BenchmarkType.MICRO)
    assert (result.type == BenchmarkType.MICRO)


def test_set_return_code():
    """Test interface BenchmarkResult.set_return_code()."""
    result = BenchmarkResult('micro', BenchmarkType.MICRO, ReturnCode.SUCCESS)
    assert (result.return_code == ReturnCode.SUCCESS)
    assert (result.result['return_code'] == [ReturnCode.SUCCESS.value])
    result.set_return_code(ReturnCode.INVALID_ARGUMENT)
    assert (result.return_code == ReturnCode.INVALID_ARGUMENT)
    assert (result.result['return_code'] == [ReturnCode.INVALID_ARGUMENT.value])
    result.set_return_code(ReturnCode.INVALID_BENCHMARK_RESULT)
    assert (result.return_code == ReturnCode.INVALID_BENCHMARK_RESULT)
    assert (result.result['return_code'] == [ReturnCode.INVALID_BENCHMARK_RESULT.value])


def test_serialize_deserialize():
    """Test serialization/deserialization and compare the results."""
    # Result with one metric.
    result = BenchmarkResult('pytorch-bert-base1', BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=2)
    result.add_result('metric1', 300, ReduceType.MAX)
    result.add_result('metric1', 200, ReduceType.MAX)
    result.add_result('metric2', 100, ReduceType.AVG)
    result.add_raw_data('metric1', [1, 2, 3], False)
    result.add_raw_data('metric1', [4, 5, 6], False)
    result.add_raw_data('metric1', [7, 8, 9], False)
    start_time = '2021-02-03 16:59:49'
    end_time = '2021-02-03 17:00:08'
    result.set_timestamp(start_time, end_time)
    result.set_benchmark_type(BenchmarkType.MICRO)

    expected = (
        '{"name": "pytorch-bert-base1", "type": "micro", "run_count": 2, "return_code": 0, '
        '"start_time": "2021-02-03 16:59:49", "end_time": "2021-02-03 17:00:08", '
        '"raw_data": {"metric1": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, '
        '"result": {"return_code": [0], "metric1": [300, 200], "metric2": [100]}, '
        '"reduce_op": {"return_code": null, "metric1": "max", "metric2": "avg"}}'
    )
    assert (result.to_string() == expected)
