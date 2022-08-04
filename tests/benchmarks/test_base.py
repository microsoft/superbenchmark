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
