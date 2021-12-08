# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for tcp-connectivity benchmark."""

import numbers
import unittest
from pathlib import Path

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.tcp_connectivity import TCPConnectivityBenchmark


class TCPConnectivityBenchmarkTest(unittest.TestCase):
    """Tests for CudaGemmFlopsBenchmark benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create hostfile just for testing.
        testdir = Path('/tmp/superbench/')
        testdir.mkdir(parents=True, exist_ok=True)
        with open('/tmp/superbench/hostfile.test', 'w') as f:
            f.write('api.github.com\n')
            f.write('localhost\n')

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        Path('/tmp/superbench/hostfile.test').unlink()

    def test_tcp_connectivity(self):
        """Test tcp-connectivity benchmark."""
        context = BenchmarkRegistry.create_benchmark_context(
            'tcp-connectivity',
            parameters='--hostfile /tmp/superbench/hostfile.test --port 80 --parallel 2',
        )
        assert (BenchmarkRegistry.is_benchmark_context_valid(context))
        benchmark = BenchmarkRegistry.launch_benchmark(context)

        # Check basic information.
        assert (benchmark)
        assert (isinstance(benchmark, TCPConnectivityBenchmark))
        assert (benchmark.name == 'tcp-connectivity')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.hostfile == '/tmp/superbench/hostfile.test')
        assert (benchmark._args.port == 80)
        assert (benchmark._args.count == 10)
        assert (benchmark._args.timeout == 1)
        assert (benchmark._args.parallel == 2)

        print(benchmark.result)
        assert (benchmark.result)

        # Check results and metrics.
        assert (benchmark.result['api.github.com_successed_count'][0] == 10)
        assert (benchmark.result['api.github.com_failed_count'][0] == 0)
        assert (benchmark.result['api.github.com_success_rate'][0] == 100.0)
        assert (isinstance(benchmark.result['api.github.com_time_min'][0], numbers.Number))
        assert (isinstance(benchmark.result['api.github.com_time_max'][0], numbers.Number))
        assert (isinstance(benchmark.result['api.github.com_time_avg'][0], numbers.Number))
        assert (isinstance(benchmark.result['localhost_successed_count'][0], numbers.Number))
        assert (isinstance(benchmark.result['localhost_failed_count'][0], numbers.Number))
        assert (isinstance(benchmark.result['localhost_time_max'][0], numbers.Number))
        assert (isinstance(benchmark.result['localhost_time_min'][0], numbers.Number))
        assert (isinstance(benchmark.result['localhost_time_avg'][0], numbers.Number))
        assert (benchmark.return_code == ReturnCode.SUCCESS)
