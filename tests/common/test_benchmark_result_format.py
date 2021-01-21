# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from superbench.common.benchmark_result import Result


def test_benchmark_result_format():
    result = Result('pytorch-bert-base')
    result.add_result('metric1', '100')
    result.add_result('metric2', '200')
    result_se = result.to_string()
    result_de = Result(**(json.loads(result_se)))
    assert(result == result_de)
