# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Device Managerment Library Utility."""

import py3nvml.py3nvml as nvml

from superbench.common.utils import logger
from superbench.common.utils import process


class DeviceManager:
    """Device management module."""
    def __init__(self):
        """Constructor."""
        nvml.nvmlInit()
        self._device_count = self.get_device_count()
        self._device_handlers = list()
        for i in range(self._device_count):
            self._device_handlers.append(nvml.nvmlDeviceGetHandleByIndex(i))

    def get_device_count(self):
        """Get the compute capability of device.

        Return:
            count (int): count of device.
        """
        return nvml.nvmlDeviceGetCount()

    def get_device_compute_capability(self):
        """Get the compute capability of device.

        Return:
            cap (float): the compute capability of device, None means failed to get the data.
        """
        try:
            cap = nvml.nvmlDeviceGetCudaComputeCapability(self._device_handlers[0])
        except Exception as err:
            logger.error('Get device compute capability failed: {}'.format(str(err)))
            return None
        return cap

    def get_device_utilization(self, idx):
        """Get the utilization of device.

        Args:
            idx (int): device index.

        Return:
            util (int): the utilization of device, None means failed to get the data.
        """
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(self._device_handlers[idx])
        except Exception as err:
            logger.error('Get device utilization failed: {}'.format(str(err)))
            return None
        return util.gpu

    def get_device_temperature(self, idx):
        """Get the temperature of device, unit: celsius.

        Args:
            idx (int): device index.

        Return:
            temp (int): the temperature of device, None means failed to get the data.
        """
        try:
            temp = nvml.nvmlDeviceGetTemperature(self._device_handlers[idx], nvml.NVML_TEMPERATURE_GPU)
        except Exception as err:
            logger.error('Get device temperature failed: {}'.format(str(err)))
            temp = None
        return temp

    def get_device_power_limit(self, idx):
        """Get the power management limit of device, unit: watt.

        Args:
            idx (int): device index.

        Return:
            temp (float): the power management limit of device, None means failed to get the data.
        """
        try:
            powerlimit = nvml.nvmlDeviceGetPowerManagementLimit(self._device_handlers[idx])
        except Exception as err:
            logger.error('Get device power limitation failed: {}'.format(str(err)))
            return None
        return int(int(powerlimit) / 1000)

    def get_device_memory(self, idx):
        """Get the memory information of device, unit: byte.

        Args:
            idx (int): device index.

        Return:
            used (float): the used device memory, None means failed to get the data.
            total (float): the total device memory, None means failed to get the data.
        """
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(self._device_handlers[idx])
        except Exception as err:
            logger.error('Get device memory failed: {}'.format(str(err)))
            return None, None
        return mem.used, mem.total

    def get_device_row_remapped_info(self, idx):
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
        output = process.run_command('nvidia-smi -i {} -q'.format(idx))
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
                    except Exception:
                        continue

                return remapped_metrics

        return None

    def get_device_ecc_error(self, idx):
        """Get the ecc error information of device.

        Args:
            idx (int): device index.

        Return:
            corrected_ecc (int)  : the count of single bit ecc error.
            uncorrected_ecc (int): the count of double bit ecc error.
        """
        corrected_ecc = 0
        uncorrected_ecc = 0
        for location_idx in range(nvml.NVML_MEMORY_LOCATION_COUNT):
            try:
                count = nvml.nvmlDeviceGetMemoryErrorCounter(
                    self._device_handlers[idx], nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, nvml.NVML_VOLATILE_ECC,
                    location_idx
                )
                corrected_ecc += count
            except nvml.NVMLError:
                pass
            except Exception as err:
                logger.error('Get device ECC information failed: {}'.format(str(err)))
                return None, None

            try:
                count = nvml.nvmlDeviceGetMemoryErrorCounter(
                    self._device_handlers[idx], nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC,
                    location_idx
                )
                uncorrected_ecc += count
            except nvml.NVMLError:
                pass
            except Exception as err:
                logger.error('Get device ECC information failed: {}'.format(str(err)))
                return None, None

        return corrected_ecc, uncorrected_ecc


device_manager = DeviceManager()
