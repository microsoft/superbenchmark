# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BenchmarkRegistry module."""

import re

from superbench.benchmarks import Platform, Framework, BenchmarkContext, BenchmarkRegistry
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

    def add_parser_auguments(self):
        """Add the specified auguments."""
        super().add_parser_auguments()

        self._parser.add_argument(
            '--lower_bound',
            type=int,
            default=0,
            metavar='',
            required=False,
            help='The lower bound for accumulation.',
        )

        self._parser.add_argument(
            '--upper_bound',
            type=int,
            default=2,
            metavar='',
            required=False,
            help='The upper bound for accumulation.',
        )

    def _benchmarking(self):
        """Implementation for benchmarking."""
        raw_data = []
        result = 0
        for i in range(self._args.lower_bound, self._args.upper_bound):
            result += i
            raw_data.append(result)

        metric = 'accumulation_result'
        self._result.add_raw_data(metric, raw_data)
        self._result.add_result(metric, result)


def test_benchmark_registration():
    """Test interface BenchmarkRegistry.register_benchmark()."""
    # Register the benchmark for all platform if use default platform.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)
    context = BenchmarkContext('accumulation', Platform.CPU)
    assert (BenchmarkRegistry.is_benchmark_registered(context))
    context = BenchmarkContext('accumulation', Platform.CUDA)
    assert (BenchmarkRegistry.is_benchmark_registered(context))
    context = BenchmarkContext('accumulation', Platform.ROCM)
    assert (BenchmarkRegistry.is_benchmark_registered(context))

    BenchmarkRegistry.clean_benchmarks()

    # Register the benchmark for CUDA platform if use platform=Platform.CUDA.
    BenchmarkRegistry.register_benchmark('accumulation-cuda', AccumulationBenchmark, platform=Platform.CUDA)
    context = BenchmarkContext('accumulation-cuda', Platform.CUDA)
    assert (BenchmarkRegistry.is_benchmark_registered(context))
    context = BenchmarkContext('accumulation-cuda', Platform.ROCM)
    assert (BenchmarkRegistry.is_benchmark_registered(context) is False)

    BenchmarkRegistry.clean_benchmarks()


def test_benchmark_context_valid():
    """Test interface BenchmarkRegistry.is_benchmark_context_valid()."""
    # Positive case.
    context = BenchmarkContext('accumulation', Platform.CPU)
    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    # Negative  case.
    context = 'context'
    assert (BenchmarkRegistry.is_benchmark_context_valid(context) is False)
    context = None
    assert (BenchmarkRegistry.is_benchmark_context_valid(context) is False)


def test_get_benchmark_name():
    """Test interface BenchmarkRegistry.get_benchmark_name()."""
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)
    BenchmarkRegistry.register_benchmark('pytorch-accumulation', AccumulationBenchmark)
    BenchmarkRegistry.register_benchmark('tf-accumulation', AccumulationBenchmark)
    BenchmarkRegistry.register_benchmark('onnx-accumulation', AccumulationBenchmark)

    # Default framework: Framework.NONE
    context = BenchmarkContext('accumulation', Platform.CPU)
    name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (name == 'accumulation')

    # Framework: Framework.PYTORCH
    context = BenchmarkContext('accumulation', Platform.CPU, framework=Framework.PYTORCH)
    name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (name == 'pytorch-accumulation')

    # Framework: Framework.TENSORFLOW
    context = BenchmarkContext('accumulation', Platform.CPU, framework=Framework.TENSORFLOW)
    name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (name == 'tf-accumulation')

    # Framework: Framework.ONNX
    context = BenchmarkContext('accumulation', Platform.CPU, framework=Framework.ONNX)
    name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (name == 'onnx-accumulation')

    BenchmarkRegistry.clean_benchmarks()


def test_check_parameters():
    """Test interface BenchmarkRegistry.check_parameters()."""
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark('accumulation', AccumulationBenchmark)

    # Positive case.
    context = BenchmarkContext('accumulation', Platform.CPU, parameters='--lower_bound=1')
    assert (BenchmarkRegistry.check_parameters(context))

    # Negative  case.
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
  --run_count     The run count of benchmark.
  --duration      The elapsed time of benchmark.
  --lower_bound   The lower bound for accumulation.
  --upper_bound   The upper bound for accumulation."""

    assert (settings == expected)

    BenchmarkRegistry.clean_benchmarks()


def test_launch_benchmark():
    """Test interface BenchmarkRegistry.launch_benchmark()."""
    # Register benchmarks for testing.
    BenchmarkRegistry.register_benchmark(
        'accumulation', AccumulationBenchmark, parameters='--upper_bound=5', platform=Platform.CPU
    )
    BenchmarkRegistry.register_benchmark(
        'tf-accumulation', AccumulationBenchmark, parameters='--upper_bound=5', platform=Platform.CPU
    )

    # Launch benchmark.
    context = BenchmarkContext(
        'accumulation', Platform.CPU, parameters='--lower_bound=1', framework=Framework.TENSORFLOW
    )

    if BenchmarkRegistry.check_parameters(context):
        result = BenchmarkRegistry.launch_benchmark(context)
        # replace the timestamp as "0"
        result = re.sub(r'\"\d+-\d+-\d+ \d+:\d+:\d+\"', '\"0\"', result)
        expected = (
            '{"name": "tf-accumulation", "type": "micro", "run_count": 1, '
            '"return_code": 0, "start_time": "0", "end_time": "0", '
            '"raw_data": {"accumulation_result": [[1, 3, 6, 10]]}, '
            '"result": {"accumulation_result": [10]}}'
        )
        assert (result == expected)

    # Launch benchmark with overrided parameters
    context = BenchmarkContext('accumulation', Platform.CPU, parameters='--lower_bound=1 --upper_bound=4')
    if BenchmarkRegistry.check_parameters(context):
        result = BenchmarkRegistry.launch_benchmark(context)
        # replace the timestamp as "0"
        result = re.sub(r'\"\d+-\d+-\d+ \d+:\d+:\d+\"', '\"0\"', result)
        expected = (
            '{"name": "accumulation", "type": "micro", "run_count": 1, '
            '"return_code": 0, "start_time": "0", "end_time": "0", '
            '"raw_data": {"accumulation_result": [[1, 3, 6]]}, '
            '"result": {"accumulation_result": [6]}}'
        )
        assert (result == expected)

    # Failed to launch benchmark.
    context = BenchmarkContext(
        'accumulation', Platform.CPU, parameters='--lower_bound=1 --upper_bound=4', framework=Framework.PYTORCH
    )
    assert (BenchmarkRegistry.check_parameters(context) is False)

    BenchmarkRegistry.clean_benchmarks()
