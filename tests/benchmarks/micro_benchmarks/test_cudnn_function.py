# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for cudnn-functions benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.cuda_test
def test_cudnn_functions():
    """Test cudnn-function benchmark."""
    # Test for default configuration
    context = BenchmarkRegistry.create_benchmark_context(
        'cudnn-function', platform=Platform.CUDA, parameters='--num_warmup 10 --num_steps 10 --num_in_step 100'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'cudnn-function')
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

    assert (18 <= len(benchmark.result))
    for metric in list(benchmark.result.keys()):
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
        if metric != 'return_code':
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)

    # Test for custom configuration
    custom_config_str = '{"algo":0,"arrayLength":2,"convType":0,"dilationA":[1,1],"filterStrideA":[1,1],' \
        + '"filterDims":[32,128,3,3],"inputDims":[32,128,14,14],"inputStride":[25088,196,14,1],"inputType":0,'\
        + '"mode":1,"name":"cudnnConvolutionBackwardFilter","outputDims":[32,32,14,14],'\
        + '"outputStride":[6272,196,14,1],"padA":[1,1],"tensorOp":false}'

    context = BenchmarkRegistry.create_benchmark_context(
        'cudnn-function',
        platform=Platform.CUDA,
        parameters=f"--num_warmup 10 --num_steps 10 --num_in_step 100 --config_json_str '{custom_config_str}'"
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'cudnn-function')
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
    custom_config_str2 = '{"algo":1,"arrayLength":2,"convType":0,"dilationA":[1,1],"filterStrideA":[1,1],' \
        + '"filterDims":[32,128,3,3],"inputDims":[32,32,14,14],"inputStride":[6272, 196, 14, 1],"inputType":2,'\
        + '"mode":1,"name":"cudnnConvolutionBackwardData","outputDims":[32, 128, 14, 14],'\
        + '"outputStride":[25088, 196, 14, 1],"padA":[1,1],"tensorOp":true}'

    context = BenchmarkRegistry.create_benchmark_context(
        'cudnn-function',
        platform=Platform.CUDA,
        parameters='--num_warmup 10 --num_steps 10 --num_in_step 100 --config_json_str ' +
        f"'{custom_config_str}' '{custom_config_str2}'"
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'cudnn-function')
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

    # Test for auto_algo parameter
    context = BenchmarkRegistry.create_benchmark_context(
        'cudnn-function',
        platform=Platform.CUDA,
        parameters='--num_warmup 10 --num_steps 10 --num_in_step 100 --enable_auto_algo'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark._args.enable_auto_algo is True)

    assert (benchmark.return_code == ReturnCode.SUCCESS)

    assert (18 + benchmark.default_metric_count == len(benchmark.result))
    for metric in list(benchmark.result.keys()):
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
        if metric != 'return_code':
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
