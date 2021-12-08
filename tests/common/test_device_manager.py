# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvidia_helper module."""

import numbers
from unittest import mock

from tests.helper import decorator
from superbench.common.utils import device_manager as dm


@decorator.cuda_test
@mock.patch('superbench.common.utils.process.run_command')
def test_nvidia_helper_utils(mock_run_command):
    """Test util functions of nvidia_helper."""
    assert (isinstance(dm.device_manager.get_device_count(), numbers.Number))
    assert (isinstance(dm.device_manager.get_device_compute_capability(), numbers.Number))
    assert (isinstance(dm.device_manager.get_device_utilization(0), numbers.Number))
    assert (isinstance(dm.device_manager.get_device_temperature(0), numbers.Number))
    assert (isinstance(dm.device_manager.get_device_power_limit(0), numbers.Number))

    used_mem, total_mem = dm.device_manager.get_device_memory(0)
    assert (isinstance(used_mem, numbers.Number) and isinstance(total_mem, numbers.Number))

    corrected_ecc, uncorrected_ecc = dm.device_manager.get_device_ecc_error(0)
    assert (isinstance(corrected_ecc, numbers.Number) and isinstance(uncorrected_ecc, numbers.Number))
    mock_run_command.return_value.returncode = 0
    mock_run_command.return_value.stdout = """
        Remapped Rows
            Correctable Error                 : 0
            Uncorrectable Error               : 0
            Pending                           : No
            Remapping Failure Occurred        : No
            Bank Remap Availability Histogram
                Max                           : 640 bank(s)
                High                          : 0 bank(s)
                Partial                       : 0 bank(s)
                Low                           : 0 bank(s)
                None                          : 0 bank(s)
        Temperature
            GPU Current Temp                  : 36 C
    """
    gpu_remapped_info = dm.device_manager.get_device_row_remapped_info(0)
    expected = {
        'gpu_remap_correctable_error': 0,
        'gpu_remap_uncorrectable_error': 0,
        'gpu_remap_max': 640,
        'gpu_remap_high': 0,
        'gpu_remap_partial': 0,
        'gpu_remap_low': 0,
        'gpu_remap_none': 0
    }
    assert (gpu_remapped_info == expected)
