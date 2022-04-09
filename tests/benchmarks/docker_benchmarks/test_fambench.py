# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for FAMBench modules."""

from types import SimpleNamespace

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, Platform, ReturnCode
from superbench.benchmarks.result import BenchmarkResult


def test_fambench():
    """Test FAMBench benchmarks."""
    benchmark_name = 'fambench'
    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
    assert (benchmark_class)
    benchmark = benchmark_class(benchmark_name)
    assert (benchmark._benchmark_type == BenchmarkType.DOCKER)
    assert (benchmark._image_uri == 'superbench/benchmark:cuda11.1.1-fambench')
    assert (benchmark._container_name == 'fambench-benchmarks')
    assert (benchmark._entrypoint == '/workspace/FAMBench/benchmarks/run_all_benchmarks.sh')
    assert (benchmark._cmd is None)
    benchmark._result = BenchmarkResult(benchmark._name, benchmark._benchmark_type, ReturnCode.SUCCESS)
    benchmark._args = SimpleNamespace(log_raw_data=False)

    raw_output = """
benchmark implementation mode config score units batch_latency_95_sec
DLRM OOTB eval tiny 152.800399 ex/s 0.515052
DLRM OOTB train tiny 35.483686 ex/s None
DLRM UBENCH train linear_[(2,2,2,2,2)] 3.679281e-07 TF/s None
XLMR OOTB eval default-config 1.015586 ex/s 16.463461
"""
    assert (benchmark._process_raw_result(0, raw_output))
    assert (benchmark.result['dlrm_ootb_eval_tiny_ex_s'][0] == 152.800399)
    assert (benchmark.result['dlrm_ootb_train_tiny_ex_s'][0] == 35.483686)
    assert (benchmark.result['dlrm_ubench_train_linear_[(2,2,2,2,2)]_tf_s'][0] == 3.679281e-07)
    assert (benchmark.result['xlmr_ootb_eval_default_config_ex_s'][0] == 1.015586)
