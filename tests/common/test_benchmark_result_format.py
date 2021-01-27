# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superbench.benchmarks import BenchmarkResult


def do_serialize_deserialize(result):
    result_se = result.to_string()
    result_de = BenchmarkResult.from_string(result_se)
    assert(result == result_de)


def test_benchmark_result_format():
    result = BenchmarkResult('pytorch-bert-base1')
    result.add_result('metric1', '300')
    do_serialize_deserialize(result)

    result = BenchmarkResult('pytorch-bert-base2')
    result.add_result('metric1', '100')
    result.add_result('metric2', '200')
    do_serialize_deserialize(result)

    result_de = BenchmarkResult.from_string('{}')
    assert(result_de is None)
