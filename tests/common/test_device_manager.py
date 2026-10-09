# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the device_manager module across NVIDIA and AMD backends."""

import numbers
from unittest import mock

from tests.helper import decorator
from superbench.common.utils import device_manager as dm

_DM_MODULE = 'superbench.common.utils.device_manager'


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


def test_amd_get_device_power_microwatts_converted():
    """average_socket_power reported in µW -> converted to watts.

    Verifies the unit handling is symmetric with get_device_power_limit so the
    monitor record's gpu_power and gpu_power_limit cannot drift by 1e6.
    """
    manager = _make_amd_manager()
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_power_info.return_value = {
        'average_socket_power': 350_000_000,    # 350 W in µW
        'current_socket_power': 360_000_000,
        'power_limit': 750_000_000,
    }
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        assert manager.get_device_power(0) == 350
        assert manager.get_device_power_limit(0) == 750


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


def test_amd_device_manager_lifecycle():
    """__init__ calls amdsmi_init/get_processor_handles; __del__ tolerates failures.

    Lifecycle is important: a regression in __del__ would surface as noisy
    NameError / AttributeError messages in benchmark logs at interpreter shutdown.
    """
    rocml_mock = mock.Mock()
    rocml_mock.amdsmi_get_processor_handles.return_value = ['h0', 'h1']
    with mock.patch(f'{_DM_MODULE}.rocml', rocml_mock, create=True):
        manager = dm.AmdDeviceManager()
        rocml_mock.amdsmi_init.assert_called_once()
        assert manager.get_device_count() == 2
        manager.__del__()
        rocml_mock.amdsmi_shut_down.assert_called_once()

    # Simulate the destructor running when amdsmi has been torn down (e.g.,
    # interpreter shutdown). It must swallow the error rather than raise.
    manager2 = dm.AmdDeviceManager.__new__(dm.AmdDeviceManager)
    manager2._device_handlers = []
    bad_rocml = mock.Mock()
    bad_rocml.amdsmi_shut_down.side_effect = RuntimeError('rocm gone')
    with mock.patch(f'{_DM_MODULE}.rocml', bad_rocml, create=True):
        manager2.__del__()    # must not raise
