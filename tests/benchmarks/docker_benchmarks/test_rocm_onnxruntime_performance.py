# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RocmOnnxRuntimeModelBenchmark modules."""

from types import SimpleNamespace

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, Platform, ReturnCode
from superbench.benchmarks.result import BenchmarkResult


def test_rocm_onnxruntime_performance():
    """Test onnxruntime model benchmark."""
    benchmark_name = 'onnxruntime-ort-models'
    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.ROCM)
    assert (benchmark_class)
    benchmark = benchmark_class(benchmark_name)
    assert (benchmark._benchmark_type == BenchmarkType.DOCKER)
    assert (benchmark._image_uri == 'superbench/benchmark:rocm4.3.1-onnxruntime1.9.0')
    assert (benchmark._container_name == 'rocm-onnxruntime-model-benchmarks')
    assert (benchmark._entrypoint == '/stage/onnxruntime-training-examples/huggingface/azureml/run_benchmark.sh')
    assert (benchmark._cmd is None)
    benchmark._result = BenchmarkResult(benchmark._name, benchmark._benchmark_type, ReturnCode.SUCCESS)
    benchmark._args = SimpleNamespace(log_raw_data=False)

    raw_output = """
__superbench__ begin bert-large-uncased ngpu=1
    "samples_per_second": 21.829
__superbench__ begin bert-large-uncased ngpu=8
    "samples_per_second": 147.181
__superbench__ begin distilbert-base-uncased ngpu=1
    "samples_per_second": 126.827
__superbench__ begin distilbert-base-uncased ngpu=8
    "samples_per_second": 966.796
__superbench__ begin gpt2 ngpu=1
    "samples_per_second": 20.46
__superbench__ begin gpt2 ngpu=8
    "samples_per_second": 151.089
__superbench__ begin facebook/bart-large ngpu=1
    "samples_per_second": 66.171
__superbench__ begin facebook/bart-large ngpu=8
    "samples_per_second": 370.343
__superbench__ begin roberta-large ngpu=1
    "samples_per_second": 37.103
__superbench__ begin roberta-large ngpu=8
    "samples_per_second": 274.455
"""
    assert (benchmark._process_raw_result(0, raw_output))
    assert (benchmark.result['bert_large_uncased_ngpu_1_throughput'][0] == 21.829)
    assert (benchmark.result['bert_large_uncased_ngpu_8_throughput'][0] == 147.181)
    assert (benchmark.result['distilbert_base_uncased_ngpu_1_throughput'][0] == 126.827)
    assert (benchmark.result['distilbert_base_uncased_ngpu_8_throughput'][0] == 966.796)
    assert (benchmark.result['gpt2_ngpu_1_throughput'][0] == 20.46)
    assert (benchmark.result['gpt2_ngpu_8_throughput'][0] == 151.089)
    assert (benchmark.result['facebook_bart_large_ngpu_1_throughput'][0] == 66.171)
    assert (benchmark.result['facebook_bart_large_ngpu_8_throughput'][0] == 370.343)
    assert (benchmark.result['roberta_large_ngpu_1_throughput'][0] == 37.103)
    assert (benchmark.result['roberta_large_ngpu_8_throughput'][0] == 274.455)
