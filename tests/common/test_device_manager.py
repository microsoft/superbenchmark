# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvidia_helper module."""

import sys
import numbers
from unittest import mock

from tests.helper import decorator
from superbench.common.utils import device_manager as dm

# Ensure the real device_manager module is loaded (bypassing the LazyImport
# proxy that ``dm`` may be) and inject a placeholder ``rocml`` so that
# AmdDeviceManager.__del__ does not raise NameError during GC of test
# instances on non-ROCm hosts.
import superbench.common.utils.device_manager as _dm_real  # noqa: E402, F401

_DM_MODULE = 'superbench.common.utils.device_manager'
if not hasattr(sys.modules[_DM_MODULE], 'rocml'):
    sys.modules[_DM_MODULE].rocml = mock.Mock()

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


def _make_amd_manager():
    """Build an AmdDeviceManager instance bypassing __init__ (no ROCm required)."""
    manager = dm.AmdDeviceManager.__new__(dm.AmdDeviceManager)
    manager._device_handlers = [mock.Mock()]
    return manager


def test_amd_get_device_power_average_supported():
    """average_socket_power is numeric -> returned as int."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {
        'average_socket_power': 123.7,
        'current_socket_power': 456,
        'power_limit': 750,
    }
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power(0) == 123


def test_amd_get_device_power_falls_back_to_current():
    """average_socket_power='N/A' -> fall back to current_socket_power."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {
        'average_socket_power': 'N/A',
        'current_socket_power': 321,
        'power_limit': 750,
    }
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power(0) == 321


def test_amd_get_device_power_both_unsupported_returns_none():
    """Both fields non-numeric -> returns None."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {
        'average_socket_power': 'N/A',
        'current_socket_power': 'N/A',
        'power_limit': 750,
    }
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power(0) is None


def test_amd_get_device_power_missing_keys_returns_none():
    """Missing keys -> None and warning logged (no exception)."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {}
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power(0) is None


def test_amd_get_device_power_limit_microwatts_converted():
    """power_limit reported in µW (e.g., 750000000) -> converted to 750 W."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {'power_limit': 750_000_000}
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power_limit(0) == 750


def test_amd_get_device_power_limit_watts_passthrough():
    """power_limit already in watts (small value) -> returned as-is."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {'power_limit': 300}
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power_limit(0) == 300


def test_amd_get_device_power_limit_non_numeric_returns_none():
    """power_limit='N/A' -> returns None."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {'power_limit': 'N/A'}
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power_limit(0) is None


def test_amd_get_device_power_limit_missing_key_returns_none():
    """Missing power_limit key -> returns None without raising."""
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {}
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power_limit(0) is None
