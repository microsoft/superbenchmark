# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench benchmark base test."""

import os
import time
import signal
import unittest
from multiprocessing import Process, Queue

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.base import Benchmark


class FooBenchmark(Benchmark):
    """Foobar benchmark for test.

    Args:
        Benchmark (Benchmark): Base Benchmark class.
    """

    def _benchmark(self):
        """Implement _benchmark method.

        Returns:
            bool: True if run benchmark successfully.
        """
        time.sleep(2)
        return True

    def test_run(self, pid_queue, rc_queue):
        """Method to test benchmark run.

        Args:
            pid_queue (Queue): Multiprocessing queue to share pid.
            rc_queue (Queue): Multiprocessing queue to share return code.
        """
        pid_queue.put(os.getpid())
        self.run()
        rc_queue.put(self.return_code)


class BenchmarkBaseTestCase(unittest.TestCase):
    """A class for benchmark base test cases."""

    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        self.benchmark = FooBenchmark('foo')
        self.benchmark._benchmark_type = BenchmarkType.MICRO
        self.pid_queue = Queue()
        self.rc_queue = Queue()

    def test_signal_handler(self):
        """Test signal handler when running benchmarks."""
        test_cases = [
            {
                'signal': None,
                'return_code': ReturnCode.SUCCESS,
            },
            {
                'signal': signal.SIGTERM,
                'return_code': ReturnCode.KILLED_BY_TIMEOUT,
            },
        ]
        for test_case in test_cases:
            with self.subTest(msg='Testing with case', test_case=test_case):
                proc = Process(target=self.benchmark.test_run, args=(
                    self.pid_queue,
                    self.rc_queue,
                ))
                proc.start()
                proc_pid = self.pid_queue.get(block=True, timeout=3)
                if test_case['signal']:
                    killer = Process(target=os.kill, args=(proc_pid, test_case['signal']))
                    killer.start()
                    killer.join()
                proc.join()
                self.assertEqual(self.rc_queue.get(block=True, timeout=3), test_case['return_code'])

    def test_compare_log_override(self):
        """Test argument override from compare_log metadata."""

        class DummyBenchmark(Benchmark):

            def add_parser_arguments(self):
                self._parser.add_argument('--compare_log', type=str, required=False)
                self._parser.add_argument('--foo', type=int, default=1)

            def _benchmark(self):
                return True

        # Patch model_log_utils.load_model_log to return dummy metadata
        from superbench.common import model_log_utils
        orig_load = model_log_utils.load_model_log
        model_log_utils.load_model_log = lambda path: {'metadata': {'foo': 42}}
        try:
            bench = DummyBenchmark('dummy', parameters='--compare_log dummy_path')
            bench._benchmark_type = BenchmarkType.MICRO
            bench.add_parser_arguments()
            ret, args, unknown = bench.parse_args()
            assert ret
            assert args.foo == 42
        finally:
            model_log_utils.load_model_log = orig_load
