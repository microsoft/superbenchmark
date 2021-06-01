# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Nvidia Utility."""

import py3nvml.py3nvml as nvml


def get_device_compute_capability():
    """Get the compute capability of device.

    Return:
        cap (float): the compute capability of device, None means no device found.
    """
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    if device_count == 0:
        return None

    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    cap = nvml.nvmlDeviceGetCudaComputeCapability(handle)
    return cap
