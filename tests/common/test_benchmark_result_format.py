# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superbench.common.benchmark_result import Result


def test_benchmark_result_format():
    result = Result('pytorch-bert-base1')
    result.add_result('metric1', '300')
    result_se = result.to_string()
    result_de = Result.from_string(result_se)
    assert(result == result_de)

    result = Result('pytorch-bert-base2')
    result.add_result('metric1', '100')
    result.add_result('metric2', '200')
    result_se = result.to_string()
    result_de = Result.from_string(result_se)
    assert(result == result_de)

    result_de = Result.from_string('{}')
    assert(result_de is None)
