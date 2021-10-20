# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Nvidia Utility."""

import py3nvml.py3nvml as nvml


def get_device_count():
    """Get number of devices.

    Return:
        device_count (int): number of devices available.
    """
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    nvml.nvmlShutdown()
    return device_count

def get_device_compute_capability():
    """Get the compute capability of device.

    Return:
        cap (float): the compute capability of device, None means no device found.
    """
    device_count = get_device_count()
    if device_count == 0:
        return None

    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    cap = nvml.nvmlDeviceGetCudaComputeCapability(handle)
    nvml.nvmlShutdown()
    return cap
