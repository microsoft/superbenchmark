# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Monitor module."""

import numbers
import tempfile
import unittest
import shutil
import pathlib

from tests.helper import decorator
from superbench.monitor import Monitor
from superbench.monitor import MonitorRecord


class MonitorTestCase(unittest.TestCase):
    """A class for Monitor test cases."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        self.sb_output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        shutil.rmtree(self.sb_output_dir)

    @decorator.cuda_test
    def test_monitor(self):
        """Test the module Monitor."""
        log_file = pathlib.Path(self.sb_output_dir) / 'monitor.log'
        monitor = Monitor(None, 1, 10, str(log_file))
        monitor._Monitor__preprocess()
        record = MonitorRecord()
        monitor._Monitor__sample_host_metrics(record)
        assert (isinstance(record.cpu_usage, numbers.Number))
        assert (record.net_receive)
        assert (record.net_transmit)
        for key, value in record.net_receive.items():
            assert ('_receive_bw' in key)
            isinstance(value, numbers.Number)
        for key, value in record.net_transmit.items():
            assert ('_transmit_bw' in key)
            isinstance(value, numbers.Number)

        monitor._Monitor__sample_gpu_metrics(record)
        gpu_list_metrics = [
            record.gpu_usage, record.gpu_temperature, record.gpu_power, record.gpu_power_limit, record.gpu_mem_used,
            record.gpu_mem_total, record.gpu_corrected_ecc, record.gpu_uncorrected_ecc
        ]
        for metric in gpu_list_metrics:
            assert (metric)
            for value in metric:
                isinstance(value, numbers.Number)
