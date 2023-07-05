# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for directx gpu device module."""

from superbench.common.devices.gpu import GPU
from tests.helper import decorator


@decorator.directx_test
def test_directx_gpu():
    """Test DirectX GPU device."""
    gpu = GPU()
    gpu.get_vendor()
    assert (gpu.vendor == 'nvidia-graphics' or gpu.vendor == 'amd-graphics')
