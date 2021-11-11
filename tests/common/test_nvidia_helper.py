# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvidia_helper module."""

import numbers

from tests.helper import decorator


@decorator.cuda_test
def test_nvidia_helper_utils():
    """Test util functions of nvidia_helper."""
    from superbench.common.utils import nv_helper
    assert (isinstance(nv_helper.get_device_count(), numbers.Number))
    assert (isinstance(nv_helper.get_device_compute_capability(), numbers.Number))
    assert (isinstance(nv_helper.get_device_utilization(0), numbers.Number))
    assert (isinstance(nv_helper.get_device_temperature(0), numbers.Number))
    assert (isinstance(nv_helper.get_device_power_limit(0), numbers.Number))
    used_mem, total_mem = nv_helper.get_device_memory(0)
    assert (isinstance(used_mem, numbers.Number) and isinstance(total_mem, numbers.Number))
    sbecc, dbecc = nv_helper.get_device_ecc_error(0)
    assert (isinstance(sbecc, numbers.Number) and isinstance(dbecc, numbers.Number))
