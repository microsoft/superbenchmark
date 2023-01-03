# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

import re

from superbench.benchmarks import Platform, Framework, BenchmarkType, BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmark
from superbench.benchmarks.micro_benchmarks.sharding_matmul import ShardingMode


class AccumulationBenchmark(MicroBenchmark):
    """Benchmark that do accumulation from lower_bound to upper_bound."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name: benchmark name.
            parameters: benchmark parameters.
        """
        super().__init__(name, parameters)

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--lower_bound',
            type=int,
            required=True,
            help='The lower bound for accumulation.',
        )

        self._parser.add_argument(
            '--upper_bound',
            type=int,
            default=2,
            required=False,
            help='The upper bound for accumulation.',
        )

    def _benchmark(self):
        """Implementation for benchmarking."""
        raw_data = []
        result = 0
        for i in range(self._args.lower_bound, self._args.upper_bound):
            result += i
            raw_data.append(str(result))

        metric = 'accumulation_result'
        self._result.add_raw_data(metric, ','.join(raw_data), self._args.log_raw_data)
        self._result.add_result(metric, result)

        return True


def test_register_benchmark():
    """Test interface BenchmarkRegistry.register_benchmark()."""
    # Register the benchmark for all platform if use default platform.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)
    for platform in Platform:
        context = BenchmarkRegistry.create_benchmark_context('accumulation', platform=platform)
        assert (BenchmarkRegistry.is_benchmark_registered(context))

    # Register the benchmark for CUDA platform if use platform=Platform.CUDA.
    BenchmarkRegistry.register_benchmark('accumulation-cuda', AccumulationBenchmark, platform=Platform.CUDA)
    context = BenchmarkRegistry.create_benchmark_context('accumulation-cuda', platform=Platform.CUDA)
    assert (BenchmarkRegistry.is_benchmark_registered(context))
    context = BenchmarkRegistry.create_benchmark_context('accumulation-cuda', platform=Platform.ROCM)
    assert (BenchmarkRegistry.is_benchmark_registered(context) is False)


def test_is_benchmark_context_valid():
    """Test interface BenchmarkRegistry.is_benchmark_context_valid()."""
    # Positive case.
    context = BenchmarkRegistry.create_benchmark_context('accumulation', platform=Platform.CPU)
    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    # Negative case.
    context = 'context'
    assert (BenchmarkRegistry.is_benchmark_context_valid(context) is False)
    context = None
    assert (BenchmarkRegistry.is_benchmark_context_valid(context) is False)


def test_get_benchmark_name():
    """Test interface BenchmarkRegistry.get_benchmark_name()."""
    # Register benchmarks for testing.
    benchmark_names = ['accumulation', 'pytorch-accumulation', 'tf1-accumulation', 'onnxruntime-accumulation']
    for name in benchmark_names:
        BenchmarkRegistry.register_benchmark(name, AccumulationBenchmark)

    # Test benchmark name for different Frameworks.
    benchmark_frameworks = [Framework.NONE, Framework.PYTORCH, Framework.TENSORFLOW1, Framework.ONNXRUNTIME]
    for i in range(len(benchmark_names)):
        context = BenchmarkRegistry.create_benchmark_context(
            'accumulation', platform=Platform.CPU, framework=benchmark_frameworks[i]
        )
        name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
        assert (name == benchmark_names[i])


def test_get_benchmark_configurable_settings():
    """Test BenchmarkRegistry interface.

    BenchmarkRegistry.get_benchmark_configurable_settings().
    """
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)

    context = BenchmarkRegistry.create_benchmark_context('accumulation', platform=Platform.CPU)
    settings = BenchmarkRegistry.get_benchmark_configurable_settings(context)

    expected = """optional arguments:
  --duration int     The elapsed time of benchmark in seconds.
  --log_flushing     Real-time log flushing.
  --log_raw_data     Log raw data into file instead of saving it into result
                     object.
  --lower_bound int  The lower bound for accumulation.
  --run_count int    The run count of benchmark.
  --upper_bound int  The upper bound for accumulation."""
    assert (settings == expected)


def test_launch_benchmark():
    """Test interface BenchmarkRegistry.launch_benchmark()."""
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark(
        'accumulation', AccumulationBenchmark, parameters='--upper_bound 5', platform=Platform.CPU
    )

    # Launch benchmark.
    context = BenchmarkRegistry.create_benchmark_context(
        'accumulation', platform=Platform.CPU, parameters='--lower_bound 1'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark)
    assert (benchmark.name == 'accumulation')
    assert (benchmark.type == BenchmarkType.MICRO)
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (benchmark.raw_data == {'accumulation_result': ['1,3,6,10']})
    assert (benchmark.result == {'return_code': [0], 'accumulation_result': [10]})

    # Replace the timestamp as null.
    result = re.sub(r'\"\d+-\d+-\d+ \d+:\d+:\d+\"', 'null', benchmark.serialized_result)
    expected = (
        '{"name": "accumulation", "type": "micro", "run_count": 1, '
        '"return_code": 0, "start_time": null, "end_time": null, '
        '"raw_data": {"accumulation_result": ["1,3,6,10"]}, '
        '"result": {"return_code": [0], "accumulation_result": [10]}, '
        '"reduce_op": {"return_code": null, "accumulation_result": null}}'
    )
    assert (result == expected)

    # Launch benchmark with overridden parameters.
    context = BenchmarkRegistry.create_benchmark_context(
        'accumulation', platform=Platform.CPU, parameters='--lower_bound 1 --upper_bound 4'
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark)
    assert (benchmark.name == 'accumulation')
    assert (benchmark.type == BenchmarkType.MICRO)
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (benchmark.raw_data == {'accumulation_result': ['1,3,6']})
    assert (benchmark.result == {'return_code': [0], 'accumulation_result': [6]})

    # Replace the timestamp as null.
    result = re.sub(r'\"\d+-\d+-\d+ \d+:\d+:\d+\"', 'null', benchmark.serialized_result)
    expected = (
        '{"name": "accumulation", "type": "micro", "run_count": 1, '
        '"return_code": 0, "start_time": null, "end_time": null, '
        '"raw_data": {"accumulation_result": ["1,3,6"]}, '
        '"result": {"return_code": [0], "accumulation_result": [6]}, '
        '"reduce_op": {"return_code": null, "accumulation_result": null}}'
    )
    assert (result == expected)

    # Failed to launch benchmark due to 'benchmark not found'.
    context = BenchmarkRegistry.create_benchmark_context(
        'accumulation-fail', Platform.CPU, parameters='--lower_bound 1 --upper_bound 4', framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark is None)

    # Failed to launch benchmark due to 'unknown arguments'.
    context = BenchmarkRegistry.create_benchmark_context(
        'accumulation', platform=Platform.CPU, parameters='--lower_bound 1 --test 4'
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark)
    assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

    # Failed to launch benchmark due to 'invalid arguments'.
    context = BenchmarkRegistry.create_benchmark_context(
        'accumulation', platform=Platform.CPU, parameters='--lower_bound 1 --upper_bound x'
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark)
    assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)


def test_get_all_benchmark_predefine_settings():
    """Test interface BenchmarkRegistry.get_all_benchmark_predefine_settings()."""
    benchmark_params = BenchmarkRegistry.get_all_benchmark_predefine_settings()

    # Choose benchmark 'pytorch-sharding-matmul' for testing.
    benchmark_name = 'pytorch-sharding-matmul'
    assert (benchmark_name in benchmark_params)
    assert (benchmark_params[benchmark_name]['run_count'] == 1)
    assert (benchmark_params[benchmark_name]['duration'] == 0)
    assert (benchmark_params[benchmark_name]['n'] == 12288)
    assert (benchmark_params[benchmark_name]['k'] == 12288)
    assert (benchmark_params[benchmark_name]['m'] == 16000)
    assert (benchmark_params[benchmark_name]['mode'] == [ShardingMode.ALLREDUCE, ShardingMode.ALLGATHER])
    assert (benchmark_params[benchmark_name]['num_warmup'] == 10)
    assert (benchmark_params[benchmark_name]['num_steps'] == 500)
