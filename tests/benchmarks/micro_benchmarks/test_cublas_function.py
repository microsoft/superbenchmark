# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for cublas-functions benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.cuda_test
def test_cublas_functions():
    """Test cublas-function benchmark."""
    # Test for default configuration
    context = BenchmarkRegistry.create_benchmark_context(
        'cublas-function', platform=Platform.CUDA, parameters='--num_warmup 10 --num_steps 10 --num_in_step 100'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'cublas-function')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.num_warmup == 10)
    assert (benchmark._args.num_steps == 10)
    assert (benchmark._args.num_in_step == 100)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output_0' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_0']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_0'][0], str))

    assert (19 <= len(benchmark.result))
    for metric in list(benchmark.result.keys()):
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
        if metric != 'return_code':
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)

    # Test for custom configuration
    custom_config_str = '{"name":"cublasCgemm","m":512,"n":512,"k":32,"transa":1,"transb":0}'
    context = BenchmarkRegistry.create_benchmark_context(
        'cublas-function',
        platform=Platform.CUDA,
        parameters=f"--num_warmup 10 --num_steps 10 --num_in_step 100 --config_json_str '{custom_config_str}'"
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'cublas-function')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.num_warmup == 10)
    assert (benchmark._args.num_steps == 10)
    assert (benchmark._args.num_in_step == 100)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output_0' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_0']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_0'][0], str))

    assert (1 + benchmark.default_metric_count == len(benchmark.result))
    for metric in list(benchmark.result.keys()):
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
        if metric != 'return_code':
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)

    # Test for custom list configuration
    custom_config_str2 = '{"name":"cublasCgemm3mStridedBatched","m":64,"n":32,"k":3,' + \
        '"transa":0,"transb":1,"batchCount":544}'
    context = BenchmarkRegistry.create_benchmark_context(
        'cublas-function',
        platform=Platform.CUDA,
        parameters='--num_warmup 10 --num_steps 10 --num_in_step 100 --config_json_str ' +
        f"'{custom_config_str}' '{custom_config_str2}'"
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'cublas-function')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.num_warmup == 10)
    assert (benchmark._args.num_steps == 10)
    assert (benchmark._args.num_in_step == 100)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output_0' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_0']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_0'][0], str))

    assert (2 + benchmark.default_metric_count == len(benchmark.result))
    for metric in list(benchmark.result.keys()):
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
        if metric != 'return_code':
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)


@decorator.cuda_test
def test_cublas_functions_correctness():
    """Test cublas-function correctness check benchmark."""
    # Test for correctness check
    context = BenchmarkRegistry.create_benchmark_context(
        'cublas-function',
        platform=Platform.CUDA,
        parameters='--num_warmup 1 --num_steps 1 --num_in_step 1 --correctness'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'cublas-function')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.correctness)

    # Check results and metrics.
    assert (1 + 3 * (len(benchmark._CublasBenchmark__default_params_dict_list)) == len(benchmark.result))
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for metric in list(benchmark.result.keys()):
        if 'correctness' in metric or 'error_rate' in metric:
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))
