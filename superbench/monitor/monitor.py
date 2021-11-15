# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Monitor class."""

import os
import time
import glob
import sched
import multiprocessing

from superbench.common.utils import logger, run_command
from superbench.common.utils import device_manager as dm
from superbench.monitor.record import MonitorRecord


class Monitor(multiprocessing.Process):
    """The monitor class to collect system metrics periodically."""
    def __init__(self, container_name, sample_duration, sample_freq, output_file):
        """Constructor.

        Args:
            container_name (str): container name that need to monitor, None means the current env.
            sample_duration (int): calculate the average metirc during sample_duration seconds.
            sample_freq (int): do sampling every sample_freq seconds.
            output_file (str): output file in jsonline format.
        """
        multiprocessing.Process.__init__(self)
        self._container_name = container_name
        self._sample_duration = sample_duration
        self._sample_freq = sample_freq
        self._output_file = output_file

        self._scheduler = sched.scheduler(time.time, time.sleep)
        self._running = multiprocessing.Value('i', 0)

        self._online_cpus = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
        self._unit_MiByte = 1024 * 1024 * 1.0

    def _preprocess(self):
        """Preprocess/preparation operations before the monitoring.

        Return:
            True if _preprocess() succeed.
        """
        if self._container_name is not None:
            output = run_command('docker ps -qf name={}'.format(self._container_name))
            if output.returncode != 0:
                logger.error(
                    'Failed to get the container id - container name: {}, error message: {}'.format(
                        self._container_name, output.stderr
                    )
                )
                return False
            container_id = output.stdout

            output = run_command('docker inspect -f {{.State.Pid}} {}'.format(container_id))
            if output.returncode != 0:
                logger.error(
                    'Failed to get the container pid - container id: {}, error message: {}'.format(
                        container_id, output.stderr
                    )
                )
                return False
            container_pid = output.stdout

            try:
                self._cpu_file = glob.glob('/sys/fs/cgroup/cpuacct/docker/{}*/cpuacct.stat'.format(container_id))[0]
                self._mem_file = glob.glob(
                    '/sys/fs/cgroup/memory/docker/{}*/memory.usage_in_bytes'.format(container_id)
                )[0]
                self._net_file = '/proc/{}/net/dev'.format(container_pid)
            except BaseException as e:
                logger.error(
                    'Faild to get the cpu/mem/net file - container: {}, error message: {}'.format(
                        self._container_name, str(e)
                    )
                )
                return False
        else:
            self._cpu_file = '/sys/fs/cgroup/cpuacct/cpuacct.stat'
            self._mem_file = '/sys/fs/cgroup/memory/memory.usage_in_bytes'
            self._net_file = '/proc/net/dev'

        return True

    def run(self):
        """Method representing the process’s activity.

        Return:
            True if launching the process succeed.
        """
        try:
            logger.info('Start monitoring.')
            self._running.value = 1
            self._output_handler = open(self._output_file, 'a')
            self._sample()
            self._scheduler.run()
        except BaseException as e:
            logger.error('Failed to launch the monitor process - error message: {}'.format(str(e)))
            self._running.value = 0
            return False

        return True

    def stop(self):
        """Method stopping the process’s activity."""
        self._running.value = 0
        list(map(self._scheduler.cancel, self._scheduler.queue))
        self.join()
        self._output_handler.close()

    def _sample(self):
        """Method sampling system metrics."""
        if self._running.value == 1:
            self._scheduler.enter(self._sample_freq, 1, self._sample, ())
            # Sampling
            record = MonitorRecord()
            self._sample_host_metrics(record)
            self._sample_gpu_metrics(record)
            self._output_handler.write('{}\n'.format(record.to_string))

    def _sample_host_metrics(self, record):
        """Method sampling the host metrics.

        Args:
            record (MonitorRecord): record instance to save the metrics.
        """
        # First round of capturing.
        system_ticks_s = self._get_total_cpu_ticks()
        container_ticks_s = self._get_process_cpu_ticks()
        start_time = time.time()
        net_bytes_s = self._get_network_bytes()

        time.sleep(self._sample_duration)

        # Second round of capturing.
        system_ticks_e = self._get_total_cpu_ticks()
        container_ticks_e = self._get_process_cpu_ticks()
        end_time = time.time()
        net_bytes_e = self._get_network_bytes()

        # Calculate CPU usage.
        cpu_usage = (container_ticks_e -
                     container_ticks_s) * 1.0 / (system_ticks_e - system_ticks_s) * self._online_cpus * 100
        record.cpu_usage = cpu_usage

        # Calculate network bandwidth.
        net_receive = dict()
        net_transmit = dict()
        for device in net_bytes_s:
            net_receive[
                '{}_receive_bw'.format(device)
            ] = ((net_bytes_e[device][0] - net_bytes_s[device][0]) / (end_time - start_time) / self._unit_MiByte)
            net_transmit[
                '{}_transmit_bw'.format(device)
            ] = ((net_bytes_e[device][1] - net_bytes_s[device][1]) / (end_time - start_time) / self._unit_MiByte)
        record.net_receive = net_receive
        record.net_transmit = net_transmit

    def _sample_gpu_metrics(self, record):
        """Method sampling the gpu metrics.

        Args:
            record (MonitorRecord): record instance to save the metrics.
        """
        gpu_usage = list()
        gpu_temperature = list()
        gpu_power_limit = list()
        gpu_mem_used = list()
        gpu_mem_total = list()
        gpu_corrected_ecc = list()
        gpu_uncorrected_ecc = list()
        gpu_remap_info = list()

        device_count = dm.device_manager.get_device_count()
        for i in range(device_count):
            gpu_usage.append(dm.device_manager.get_device_utilization(i))
            gpu_temperature.append(dm.device_manager.get_device_temperature(i))
            gpu_power_limit.append(dm.device_manager.get_device_power_limit(i))
            mem_used, mem_total = dm.device_manager.get_device_memory(i)
            gpu_mem_used.append(mem_used)
            gpu_mem_total.append(mem_total)
            corrected_ecc, uncorrected_ecc = dm.device_manager.get_device_ecc_error(i)
            gpu_corrected_ecc.append(corrected_ecc)
            gpu_uncorrected_ecc.append(uncorrected_ecc)
            gpu_remap_info.append(dm.device_manager.get_device_row_remapped_info(i))

        record.gpu_usage = gpu_usage
        record.gpu_temperature = gpu_temperature
        record.gpu_power_limit = gpu_power_limit
        record.gpu_mem_used = gpu_mem_used
        record.gpu_mem_total = gpu_mem_total
        record.gpu_corrected_ecc = gpu_corrected_ecc
        record.gpu_uncorrected_ecc = gpu_uncorrected_ecc
        record.gpu_remap_info = gpu_remap_info

    def _get_total_cpu_ticks(self):
        """Method to get the total cpu ticks.

        Return:
            The total cpu ticks, None means fail to get the data.
        """
        try:
            with open('/proc/stat', 'r') as f:
                for line in f.readlines():
                    if line.startswith('cpu '):
                        items = line.split()
                        total_clock_ticks = 0
                        for item in items[1:8]:
                            total_clock_ticks += int(item)
                        return total_clock_ticks
        except BaseException as e:
            logger.error('Failed to read total cpu ticks infomation - error message: {}'.format(str(e)))

        return None

    def _get_process_cpu_ticks(self):
        """Method to get the process cpu ticks.

        Return:
            The process cpu ticks, None means fail to get the data.
        """
        user_time = 0
        system_time = 0
        try:
            with open(self._cpu_file, 'r') as f:
                for line in f:
                    items = line.split()
                    if items[0] == 'user':
                        user_time = int(items[1])
                    elif items[1] == 'system':
                        system_time = int(items[1])
                return user_time + system_time
        except BaseException as e:
            logger.error('Failed to read process cpu ticks infomation - error message: {}'.format(str(e)))

        return None

    def _get_network_bytes(self):
        """Method to get the network traffic information, unit: bytes.

        Return:
            The bytes transferred on the network, None means fail to get the data.
        """
        net_info = dict()
        try:
            with open(self._net_file, 'r') as f:
                for line in f:
                    items = line.split()
                    if len(items) != 17:
                        continue
                    else:
                        receive_bytes = int(items[1])
                        transmit_bytes = int(items[9])
                        net_info[items[0].strip()[:-1]] = [receive_bytes, transmit_bytes]
            return net_info
        except BaseException as e:
            logger.error('Failed to read network traffic infomation - error message: {}'.format(str(e)))

        return None
