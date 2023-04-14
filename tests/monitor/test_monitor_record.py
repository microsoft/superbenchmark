# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for MonitorRecord module."""

import re
import json

from superbench.monitor import MonitorRecord


def test_monitor_record():
    """Test the module MonitorRecord."""
    mr = MonitorRecord()
    mr.cpu_usage = 80
    mr.mem_used = 100
    mr.mem_total = 1024
    mr.gpu_usage = [90, 80, 86, 72, 79, 81, 94, 85]
    mr.gpu_temperature = [62, 75, 69, 63, 72, 77, 80, 71]
    mr.gpu_power = [257, 290, 280, 262, 291, 284, 281, 273]
    mr.gpu_power_limit = [400, 400, 400, 350, 400, 400, 400, 400]
    mr.gpu_mem_used = [2550, 2680, 2543, 2588, 2612, 2603, 2515, 2593]
    mr.gpu_mem_total = [16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216]
    mr.gpu_corrected_ecc = [0, 0, 0, 0, 0, 0, 0, 0]
    mr.gpu_uncorrected_ecc = [0, 0, 0, 0, 0, 0, 0, 0]
    gpu_remap_info = {
        'gpu_remap_correctable_error': 0,
        'gpu_remap_uncorrectable_error': 0,
        'gpu_remap_max': 640,
        'gpu_remap_high': 0,
        'gpu_remap_partial': 0,
        'gpu_remap_low': 0,
        'gpu_remap_none': 0
    }
    gpu_remap_infos = list()
    for i in range(8):
        gpu_remap_infos.append(gpu_remap_info)
    mr.gpu_remap_info = gpu_remap_infos
    mr.net_receive = {'eth0_receive_bw': 100, 'ib0_receive_bw': 1000}
    mr.net_transmit = {'eth0_transmit_bw': 80, 'ib0_transmit_bw': 800}

    expected_record = {
        'time': '2021-11-18 06:24:00',
        'cpu_usage': 80,
        'mem_used': 100,
        'mem_total': 1024,
        'gpu_usage:0': 90,
        'gpu_usage:1': 80,
        'gpu_usage:2': 86,
        'gpu_usage:3': 72,
        'gpu_usage:4': 79,
        'gpu_usage:5': 81,
        'gpu_usage:6': 94,
        'gpu_usage:7': 85,
        'gpu_temperature:0': 62,
        'gpu_temperature:1': 75,
        'gpu_temperature:2': 69,
        'gpu_temperature:3': 63,
        'gpu_temperature:4': 72,
        'gpu_temperature:5': 77,
        'gpu_temperature:6': 80,
        'gpu_temperature:7': 71,
        'gpu_power:0': 257,
        'gpu_power:1': 290,
        'gpu_power:2': 280,
        'gpu_power:3': 262,
        'gpu_power:4': 291,
        'gpu_power:5': 284,
        'gpu_power:6': 281,
        'gpu_power:7': 273,
        'gpu_power_limit:0': 400,
        'gpu_power_limit:1': 400,
        'gpu_power_limit:2': 400,
        'gpu_power_limit:3': 350,
        'gpu_power_limit:4': 400,
        'gpu_power_limit:5': 400,
        'gpu_power_limit:6': 400,
        'gpu_power_limit:7': 400,
        'gpu_mem_used:0': 2550,
        'gpu_mem_used:1': 2680,
        'gpu_mem_used:2': 2543,
        'gpu_mem_used:3': 2588,
        'gpu_mem_used:4': 2612,
        'gpu_mem_used:5': 2603,
        'gpu_mem_used:6': 2515,
        'gpu_mem_used:7': 2593,
        'gpu_mem_total:0': 16777216,
        'gpu_mem_total:1': 16777216,
        'gpu_mem_total:2': 16777216,
        'gpu_mem_total:3': 16777216,
        'gpu_mem_total:4': 16777216,
        'gpu_mem_total:5': 16777216,
        'gpu_mem_total:6': 16777216,
        'gpu_mem_total:7': 16777216,
        'gpu_corrected_ecc:0': 0,
        'gpu_corrected_ecc:1': 0,
        'gpu_corrected_ecc:2': 0,
        'gpu_corrected_ecc:3': 0,
        'gpu_corrected_ecc:4': 0,
        'gpu_corrected_ecc:5': 0,
        'gpu_corrected_ecc:6': 0,
        'gpu_corrected_ecc:7': 0,
        'gpu_uncorrected_ecc:0': 0,
        'gpu_uncorrected_ecc:1': 0,
        'gpu_uncorrected_ecc:2': 0,
        'gpu_uncorrected_ecc:3': 0,
        'gpu_uncorrected_ecc:4': 0,
        'gpu_uncorrected_ecc:5': 0,
        'gpu_uncorrected_ecc:6': 0,
        'gpu_uncorrected_ecc:7': 0,
        'gpu_remap_correctable_error:0': 0,
        'gpu_remap_uncorrectable_error:0': 0,
        'gpu_remap_max:0': 640,
        'gpu_remap_high:0': 0,
        'gpu_remap_partial:0': 0,
        'gpu_remap_low:0': 0,
        'gpu_remap_none:0': 0,
        'gpu_remap_correctable_error:1': 0,
        'gpu_remap_uncorrectable_error:1': 0,
        'gpu_remap_max:1': 640,
        'gpu_remap_high:1': 0,
        'gpu_remap_partial:1': 0,
        'gpu_remap_low:1': 0,
        'gpu_remap_none:1': 0,
        'gpu_remap_correctable_error:2': 0,
        'gpu_remap_uncorrectable_error:2': 0,
        'gpu_remap_max:2': 640,
        'gpu_remap_high:2': 0,
        'gpu_remap_partial:2': 0,
        'gpu_remap_low:2': 0,
        'gpu_remap_none:2': 0,
        'gpu_remap_correctable_error:3': 0,
        'gpu_remap_uncorrectable_error:3': 0,
        'gpu_remap_max:3': 640,
        'gpu_remap_high:3': 0,
        'gpu_remap_partial:3': 0,
        'gpu_remap_low:3': 0,
        'gpu_remap_none:3': 0,
        'gpu_remap_correctable_error:4': 0,
        'gpu_remap_uncorrectable_error:4': 0,
        'gpu_remap_max:4': 640,
        'gpu_remap_high:4': 0,
        'gpu_remap_partial:4': 0,
        'gpu_remap_low:4': 0,
        'gpu_remap_none:4': 0,
        'gpu_remap_correctable_error:5': 0,
        'gpu_remap_uncorrectable_error:5': 0,
        'gpu_remap_max:5': 640,
        'gpu_remap_high:5': 0,
        'gpu_remap_partial:5': 0,
        'gpu_remap_low:5': 0,
        'gpu_remap_none:5': 0,
        'gpu_remap_correctable_error:6': 0,
        'gpu_remap_uncorrectable_error:6': 0,
        'gpu_remap_max:6': 640,
        'gpu_remap_high:6': 0,
        'gpu_remap_partial:6': 0,
        'gpu_remap_low:6': 0,
        'gpu_remap_none:6': 0,
        'gpu_remap_correctable_error:7': 0,
        'gpu_remap_uncorrectable_error:7': 0,
        'gpu_remap_max:7': 640,
        'gpu_remap_high:7': 0,
        'gpu_remap_partial:7': 0,
        'gpu_remap_low:7': 0,
        'gpu_remap_none:7': 0,
        'eth0_receive_bw': 100,
        'ib0_receive_bw': 1000,
        'eth0_transmit_bw': 80,
        'ib0_transmit_bw': 800
    }

    result = mr.to_string()
    result = re.sub(r'\"\d+-\d+-\d+ \d+:\d+:\d+\"', '\"2021-11-18 06:24:00\"', result)
    assert (json.loads(result) == expected_record)
