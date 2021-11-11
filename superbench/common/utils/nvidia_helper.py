# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Nvidia Utility."""

import py3nvml.py3nvml as nvml

from superbench.common.utils import logger
from superbench.common.utils import run_command

nvml.nvmlInit()


def get_device_count():
    """Get the compute capability of device.

    Return:
        count (int): count of device.
    """
    return nvml.nvmlDeviceGetCount()


def get_device_compute_capability():
    """Get the compute capability of device.

    Return:
        cap (float): the compute capability of device, None means no device found.
    """
    device_count = nvml.nvmlDeviceGetCount()
    if device_count == 0:
        return None

    handle = nvml.nvmlDeviceGetHandleByIndex(0)
    cap = nvml.nvmlDeviceGetCudaComputeCapability(handle)
    return cap


def get_device_utilization(idx):
    """Get the utilization of device.

    Args:
        idx (int): device index.

    Return:
        util (int): the utilization of device, None means failed to get the data.
    """
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
    except nvml.NVMLError as err:
        logger.error('Get device utilization failed: {}'.format(str(err)))
        return None
    return util.gpu


def get_device_temperature(idx):
    """Get the temperature of device, unit: celsius.

    Args:
        idx (int): device index.

    Return:
        temp (int): the temperature of device, None means failed to get the data.
    """
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
    except nvml.NVMLError as err:
        logger.error('Get device temperature failed: {}'.format(str(err)))
        temp = None
    return temp


def get_device_power_limit(idx):
    """Get the power management limit of device, unit: watt.

    Args:
        idx (int): device index.

    Return:
        temp (float): the power management limit of device, None means failed to get the data.
    """
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
        powerlimit = nvml.nvmlDeviceGetPowerManagementLimit(handle)
    except nvml.NVMLError as err:
        logger.error('Get device power limitation failed: {}'.format(str(err)))
        return None
    return int(int(powerlimit) / 1000)


def get_device_memory(idx):
    """Get the memory information of device, unit: byte.

    Args:
        idx (int): device index.

    Return:
        used (float): the used device memory, None means failed to get the data.
        total (float): the total device memory, None means failed to get the data.
    """
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)
    except nvml.NVMLError as err:
        logger.error('Get device memory failed: {}'.format(str(err)))
        return None, None
    return mem.used, mem.total


def get_device_row_remapped_info(idx):
    """Get the row remapped information of device.

    The command 'nvidia-smi -i idx -q' contains the following output:
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

    Args:
        idx (int): device index.

    Return:
        remapped_metrics (dict): the row remapped information, None means failed to get the data.
    """
    output = run_command('nvidia-smi -i {} -q'.format(idx))
    if output.returncode == 0:
        begin = output.stdout.find('Remapped Rows')
        end = output.stdout.find('Temperature', begin)
        if begin != -1 and end != -1:
            remapped_info = output.stdout[begin:end]
            remapped_info = remapped_info.split('\n')
            remapped_info = [item for item in remapped_info if ':' in item]
            remapped_metrics = dict()
            for item in remapped_info:
                key_value = item.split(':')
                key = 'gpu_remap_' + key_value[0].lower().strip().replace(' ', '_')
                value = key_value[1].replace('bank(s)', '').strip()
                try:
                    value = int(value)
                    remapped_metrics[key] = value
                except BaseException:
                    continue

            return remapped_metrics

    return None


def get_device_ecc_error(idx):
    """Get the ecc error information of device.

    The command 'nvidia-smi dmon -i 0 -s e -c 1' gets the following output:
        # gpu sbecc dbecc   pci
        # Idx  errs  errs  errs
            0     0     0     0

    Args:
        idx (int): device index.

    Return:
        sbecc (int): the count of single bit ecc error, None means failed to get the data.
        dbecc (int): the count of double bit ecc error, None means failed to get the data.
    """
    output = run_command('nvidia-smi dmon -i {} -s e -c 1'.format(idx))
    if output.returncode == 0:
        content = output.stdout.splitlines()
        if len(content) == 3:
            ecc = list(filter(None, content[2].split(' ')))
            if len(ecc) == 4:
                sbecc = int(ecc[1])
                dbecc = int(ecc[2])
                return sbecc, dbecc

    return None, None
