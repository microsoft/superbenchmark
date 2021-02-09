# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

import re

from superbench.benchmarks import Platform, Framework, BenchmarkType, BenchmarkContext, BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmark


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
            default=0,
            required=False,
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
        self._result.add_raw_data(metric, ','.join(raw_data))
        self._result.add_result(metric, result)

        return True


def test_register_benchmark():
    """Test interface BenchmarkRegistry.register_benchmark()."""
    # Register the benchmark for all platform if use default platform.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)
    for platform in Platform:
        context = BenchmarkContext('accumulation', platform)
        assert (BenchmarkRegistry.is_benchmark_registered(context))

    BenchmarkRegistry.clean_benchmarks()

    # Register the benchmark for CUDA platform if use platform=Platform.CUDA.
    BenchmarkRegistry.register_benchmark('accumulation-cuda', AccumulationBenchmark, platform=Platform.CUDA)
    context = BenchmarkContext('accumulation-cuda', Platform.CUDA)
    assert (BenchmarkRegistry.is_benchmark_registered(context))
    context = BenchmarkContext('accumulation-cuda', Platform.ROCM)
    assert (BenchmarkRegistry.is_benchmark_registered(context) is False)

    BenchmarkRegistry.clean_benchmarks()


def test_is_benchmark_context_valid():
    """Test interface BenchmarkRegistry.is_benchmark_context_valid()."""
    # Positive case.
    context = BenchmarkContext('accumulation', Platform.CPU)
    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    # Negative case.
    context = 'context'
    assert (BenchmarkRegistry.is_benchmark_context_valid(context) is False)
    context = None
    assert (BenchmarkRegistry.is_benchmark_context_valid(context) is False)


def test_get_benchmark_name():
    """Test interface BenchmarkRegistry.get_benchmark_name()."""
    # Register benchmarks for testing.
    benchmark_names = ['accumulation', 'pytorch-accumulation', 'tf1-accumulation', 'onnx-accumulation']
    for name in benchmark_names:
        BenchmarkRegistry.register_benchmark(name, AccumulationBenchmark)

    # Test benchmark name for different Frameworks.
    benchmark_frameworks = [Framework.NONE, Framework.PYTORCH, Framework.TENSORFLOW, Framework.ONNX]
    for i in range(len(benchmark_names)):
        context = BenchmarkContext('accumulation', Platform.CPU, framework=benchmark_frameworks[i])
        name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
        assert (name == benchmark_names[i])

    BenchmarkRegistry.clean_benchmarks()


def test_check_parameters():
    """Test interface BenchmarkRegistry.check_parameters()."""
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)

    # Positive case.
    context = BenchmarkContext('accumulation', Platform.CPU, parameters='--lower_bound=1')
    assert (BenchmarkRegistry.check_parameters(context))

    # Negative case.
    context = BenchmarkContext('accumulation', Platform.CPU, parameters='--lower=1')
    assert (BenchmarkRegistry.check_parameters(context) is False)

    BenchmarkRegistry.clean_benchmarks()


def test_get_benchmark_configurable_settings():
    """Test BenchmarkRegistry interface.

    BenchmarkRegistry.get_benchmark_configurable_settings().
    """
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)

    context = BenchmarkContext('accumulation', Platform.CPU)
    settings = BenchmarkRegistry.get_benchmark_configurable_settings(context)

    expected = """optional arguments:
  --run_count int    The run count of benchmark.
  --duration int     The elapsed time of benchmark in seconds.
  --lower_bound int  The lower bound for accumulation.
  --upper_bound int  The upper bound for accumulation."""
    assert (settings == expected)

    BenchmarkRegistry.clean_benchmarks()


def test_launch_benchmark():
    """Test interface BenchmarkRegistry.launch_benchmark()."""
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark(
        'accumulation', AccumulationBenchmark, parameters='--upper_bound=5', platform=Platform.CPU
    )

    # Launch benchmark.
    context = BenchmarkContext('accumulation', Platform.CPU, parameters='--lower_bound=1')

    if BenchmarkRegistry.check_parameters(context):
        benchmark = BenchmarkRegistry.launch_benchmark(context)
        assert (benchmark)
        assert (benchmark.name == 'accumulation')
        assert (benchmark.type == BenchmarkType.MICRO.value)
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == ReturnCode.SUCCESS.value)
        assert (benchmark.raw_data == {'accumulation_result': ['1,3,6,10']})
        assert (benchmark.result == {'accumulation_result': [10]})

        # Replace the timestamp as null.
        result = re.sub(r'\"\d+-\d+-\d+ \d+:\d+:\d+\"', 'null', benchmark.serialized_result)
        expected = (
            '{"name": "accumulation", "type": "micro", "run_count": 1, '
            '"return_code": 0, "start_time": null, "end_time": null, '
            '"raw_data": {"accumulation_result": ["1,3,6,10"]}, '
            '"result": {"accumulation_result": [10]}}'
        )
        assert (result == expected)

    # Launch benchmark with overridden parameters.
    context = BenchmarkContext('accumulation', Platform.CPU, parameters='--lower_bound=1 --upper_bound=4')
    if BenchmarkRegistry.check_parameters(context):
        benchmark = BenchmarkRegistry.launch_benchmark(context)
        assert (benchmark)
        assert (benchmark.name == 'accumulation')
        assert (benchmark.type == BenchmarkType.MICRO.value)
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == 0)
        assert (benchmark.raw_data == {'accumulation_result': ['1,3,6']})
        assert (benchmark.result == {'accumulation_result': [6]})

        # Replace the timestamp as null.
        result = re.sub(r'\"\d+-\d+-\d+ \d+:\d+:\d+\"', 'null', benchmark.serialized_result)
        expected = (
            '{"name": "accumulation", "type": "micro", "run_count": 1, '
            '"return_code": 0, "start_time": null, "end_time": null, '
            '"raw_data": {"accumulation_result": ["1,3,6"]}, '
            '"result": {"accumulation_result": [6]}}'
        )
        assert (result == expected)

    # Failed to launch benchmark.
    context = BenchmarkContext(
        'accumulation', Platform.CPU, parameters='--lower_bound=1 --upper_bound=4', framework=Framework.PYTORCH
    )
    assert (BenchmarkRegistry.check_parameters(context) is False)

    BenchmarkRegistry.clean_benchmarks()
