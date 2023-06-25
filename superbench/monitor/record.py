# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data of monitor."""

import json
import numbers
from datetime import datetime

from superbench.benchmarks import ReduceType


class MonitorRecord:
    """Record class to save all monitoring data."""
    reduce_ops = {
        'gpu_temperature': ReduceType.MAX,
        'gpu_power': ReduceType.MAX,
        'gpu_power_limit': ReduceType.MIN,
        'gpu_corrected_ecc': ReduceType.LAST,
        'gpu_uncorrected_ecc': ReduceType.LAST,
        'gpu_remap': ReduceType.LAST,
    }

    def __init__(self):
        """Constructor."""
        self.__time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        self.__cpu_usage = None
        self.__mem_used = None
        self.__mem_total = None
        self.__gpu_usage = list()
        self.__gpu_temperature = list()
        self.__gpu_power = list()
        self.__gpu_power_limit = list()
        self.__gpu_mem_used = list()
        self.__gpu_mem_total = list()
        self.__gpu_corrected_ecc = list()
        self.__gpu_uncorrected_ecc = list()
        self.__gpu_remap_info = list()
        self.__net_receive = dict()
        self.__net_transmit = dict()

    @property
    def time(self):
        """Decoration function to access __time."""
        return self.__time

    @property
    def cpu_usage(self):
        """Decoration function to access __cpu_usage."""
        return self.__cpu_usage

    @cpu_usage.setter
    def cpu_usage(self, usage):
        """Set the cpu usage.

        Args:
            usage (float): cpu usage.
        """
        self.__cpu_usage = usage

    @property
    def mem_used(self):
        """Decoration function to access __mem_used."""
        return self.__mem_used

    @mem_used.setter
    def mem_used(self, mem_used):
        """Set the used host memory, unit: MB.

        Args:
            mem_used (float): used host memory.
        """
        self.__mem_used = mem_used

    @property
    def mem_total(self):
        """Decoration function to access __mem_total."""
        return self.__mem_total

    @mem_total.setter
    def mem_total(self, mem_total):
        """Set the total host memory, unit: MB.

        Args:
            mem_total (float): total host memory.
        """
        self.__mem_total = mem_total

    @property
    def gpu_usage(self):
        """Decoration function to access __gpu_usage."""
        return self.__gpu_usage

    @gpu_usage.setter
    def gpu_usage(self, gpu_usage):
        """Set the gpu usage.

        Args:
            gpu_usage (list): list of gpu usage.
        """
        self.__gpu_usage = gpu_usage

    @property
    def gpu_temperature(self):
        """Decoration function to access __gpu_temperature."""
        return self.__gpu_temperature

    @gpu_temperature.setter
    def gpu_temperature(self, gpu_temperature):
        """Set the gpu temperature, unit: Celsius.

        Args:
            gpu_temperature (list): list of gpu temperature.
        """
        self.__gpu_temperature = gpu_temperature

    @property
    def gpu_power(self):
        """Decoration function to access __gpu_power."""
        return self.__gpu_power

    @gpu_power.setter
    def gpu_power(self, gpu_power):
        """Set the gpu realtime power, unit: Watt.

        Args:
            gpu_power(list): list of gpu realtime power.
        """
        self.__gpu_power = gpu_power

    @property
    def gpu_power_limit(self):
        """Decoration function to access __gpu_power_limit."""
        return self.__gpu_power_limit

    @gpu_power_limit.setter
    def gpu_power_limit(self, gpu_power_limit):
        """Set the gpu power limit, unit: Watt.

        Args:
            gpu_power_limit (list): list of gpu power limit.
        """
        self.__gpu_power_limit = gpu_power_limit

    @property
    def gpu_mem_used(self):
        """Decoration function to access __gpu_mem_used."""
        return self.__gpu_mem_used

    @gpu_mem_used.setter
    def gpu_mem_used(self, gpu_mem_used):
        """Set the used gpu memory, unit: MB.

        Args:
            gpu_mem_used (list): list of used gpu memory.
        """
        self.__gpu_mem_used = gpu_mem_used

    @property
    def gpu_mem_total(self):
        """Decoration function to access __gpu_mem_total."""
        return self.__gpu_mem_total

    @gpu_mem_total.setter
    def gpu_mem_total(self, gpu_mem_total):
        """Set the total gpu memory, unit: MB.

        Args:
            gpu_mem_total (list): list of total gpu memory.
        """
        self.__gpu_mem_total = gpu_mem_total

    @property
    def gpu_corrected_ecc(self):
        """Decoration function to access __gpu_corrected_ecc."""
        return self.__gpu_corrected_ecc

    @gpu_corrected_ecc.setter
    def gpu_corrected_ecc(self, gpu_corrected_ecc):
        """Set the count of corrected (single bit) ecc error.

        Args:
            gpu_corrected_ecc (list): list of gpu corrected ecc error.
        """
        self.__gpu_corrected_ecc = gpu_corrected_ecc

    @property
    def gpu_uncorrected_ecc(self):
        """Decoration function to access __gpu_uncorrected_ecc."""
        return self.__gpu_uncorrected_ecc

    @gpu_uncorrected_ecc.setter
    def gpu_uncorrected_ecc(self, gpu_uncorrected_ecc):
        """Set the count of uncorrected (double bit) ecc error.

        Args:
            gpu_uncorrected_ecc (list): list of gpu uncorrected ecc error.
        """
        self.__gpu_uncorrected_ecc = gpu_uncorrected_ecc

    @property
    def gpu_remap_info(self):
        """Decoration function to access __gpu_remap_info."""
        return self.__gpu_remap_info

    @gpu_remap_info.setter
    def gpu_remap_info(self, gpu_remap_info):
        """Set the gpu remap_info.

        Args:
            gpu_remap_info (list): list of gpu remap_info.
        """
        self.__gpu_remap_info = gpu_remap_info

    @property
    def net_receive(self):
        """Decoration function to access __net_receive."""
        return self.__net_receive

    @net_receive.setter
    def net_receive(self, net_receive):
        """Set the network receive bandwidth, unit: Bytes/s.

        Args:
            net_receive (dict): receive bandwidth for all devices.
        """
        self.__net_receive = net_receive

    @property
    def net_transmit(self):
        """Decoration function to access __net_transmit."""
        return self.__net_transmit

    @net_transmit.setter
    def net_transmit(self, net_transmit):
        """Set the network transmit bandwidth, unit: Bytes/s.

        Args:
            net_transmit (dict): transmit bandwidth for all devices.
        """
        self.__net_transmit = net_transmit

    def to_string(self):
        """Serialize the MonitorRecord object to string.

        Return:
            The serialized string of MonitorRecord object.
        """
        formatted_obj = dict()
        for key, value in self.__dict__.items():
            # The name of internal member is like '_MonitorRecord__name'.
            # For the result object return to caller, need to reformat the 'name' as key.
            formatted_key = key.split('__')[1]
            if isinstance(value, numbers.Number) or isinstance(value, str):
                formatted_obj[formatted_key] = value
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, numbers.Number):
                        formatted_obj['{}:{}'.format(formatted_key, i)] = item
                    elif isinstance(item, dict):
                        for k, v in item.items():
                            formatted_obj['{}:{}'.format(k, i)] = v
            elif isinstance(value, dict):
                for k, v in value.items():
                    formatted_obj[k] = v

        return json.dumps(formatted_obj)
