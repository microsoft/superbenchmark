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
    def __init__(self, container_name, sample_duration, sample_interval, output_file):
        """Constructor.

        Args:
            container_name (str): container name that need to monitor, None means the current env.
            sample_duration (int): calculate the average metirc during sample_duration seconds.
            sample_interval (int): do sampling every sample_interval seconds.
            output_file (str): output file in jsonline format.
        """
        multiprocessing.Process.__init__(self)
        self.__container_name = container_name
        self.__sample_duration = sample_duration
        self.__sample_interval = sample_interval
        self.__output_file = output_file

        self.__scheduler = sched.scheduler(time.time, time.sleep)
        self.__running = multiprocessing.Value('i', 0)

        self.__online_cpus = os.sysconf(os.sysconf_names['SC_NPROCESSORS_ONLN'])
        self.__unit_MiByte = 1024 * 1024 * 1.0

        self.__output_handler = open(self.__output_file, 'a')
        self.__cgroup = 1

    def __preprocess(self):
        """Preprocess/preparation operations before the monitoring.

        Return:
            True if __preprocess() succeed.
        """
        if self.__container_name is not None:
            output = run_command('docker ps -qf name={}'.format(self.__container_name))
            if output.returncode != 0:
                logger.error(
                    'Failed to get the container id - container name: {}, error message: {}'.format(
                        self.__container_name, output.stderr
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
                cpu_file_cgroup_v1 = glob.glob('/sys/fs/cgroup/cpuacct/docker/{}*/cpuacct.stat'.format(container_id))
                if len(cpu_file_cgroup_v1) > 0:
                    self._cpu_file = cpu_file_cgroup_v1[0]
                    self._mem_file = glob.glob(
                        '/sys/fs/cgroup/memory/docker/{}*/memory.usage_in_bytes'.format(container_id)
                    )[0]
                    self._net_file = '/proc/{}/net/dev'.format(container_pid)
                else:
                    self.__cgroup = 2
                    self._cpu_file = glob.glob(
                        '/sys/fs/cgroup/system.slice/docker-{}*.scope/cpu.stat'.format(container_id)
                    )[0]
                    self._mem_file = glob.glob(
                        '/sys/fs/cgroup/system.slice/docker-{}*.scope/memory.stat'.format(container_id)
                    )[0]
                    self._net_file = '/proc/net/dev'
            except BaseException as e:
                logger.error(
                    'Faild to get the cpu/mem/net file - container: {}, error message: {}'.format(
                        self.__container_name, str(e)
                    )
                )
                return False
        else:
            cpu_file_cgroup_v1 = '/sys/fs/cgroup/cpuacct/cpuacct.stat'
            if os.path.exists(cpu_file_cgroup_v1):
                self._cpu_file = cpu_file_cgroup_v1
                self._mem_file = '/sys/fs/cgroup/memory/memory.usage_in_bytes'
            else:
                self.__cgroup = 2
                self._cpu_file = '/sys/fs/cgroup/cpu.stat'
                self._mem_file = '/sys/fs/cgroup/memory.stat'
            self._net_file = '/proc/net/dev'

        return True

    def run(self):
        """Method representing the process’s activity.

        Return:
            True if launching the process succeed.
        """
        if self.__running.value == 0:
            if not self.__preprocess():
                return False

            try:
                logger.info('Start monitoring.')
                self.__running.value = 1
                self.__sample()
                self.__scheduler.run()
            except BaseException as e:
                logger.error('Failed to launch the monitor process - error message: {}'.format(str(e)))
                self.stop()
                return False
        else:
            logger.error('Monitor is still running')

        return True

    def stop(self):
        """Method stopping the process’s activity."""
        self.__running.value = 0
        list(map(self.__scheduler.cancel, self.__scheduler.queue))
        self.join()
        self.__output_handler.close()

    def __sample(self):
        """Method sampling system metrics."""
        if self.__running.value == 1:
            self.__scheduler.enter(self.__sample_interval, 1, self.__sample, ())
            # Sampling
            record = MonitorRecord()
            self.__sample_host_metrics(record)
            self.__sample_gpu_metrics(record)
            self.__output_handler.write('{}\n'.format(record.to_string()))
            self.__output_handler.flush()

    def __sample_host_metrics(self, record):
        """Method sampling the host metrics.

        Args:
            record (MonitorRecord): record instance to save the metrics.
        """
        # First round of capturing.
        system_ticks_s = self.__get_total_cpu_ticks()
        container_ticks_s = self.__get_process_cpu_ticks()
        start_time = time.time()
        net_bytes_s = self.__get_network_bytes()

        time.sleep(self.__sample_duration)

        # Second round of capturing.
        system_ticks_e = self.__get_total_cpu_ticks()
        container_ticks_e = self.__get_process_cpu_ticks()
        end_time = time.time()
        net_bytes_e = self.__get_network_bytes()

        # Calculate CPU usage.
        cpu_usage = (container_ticks_e -
                     container_ticks_s) * 1.0 / (system_ticks_e - system_ticks_s) * self.__online_cpus * 100
        record.cpu_usage = cpu_usage

        # Calculate network bandwidth.
        net_receive = dict()
        net_transmit = dict()
        for device in net_bytes_s:
            net_receive[
                '{}_receive_bw'.format(device)
            ] = ((net_bytes_e[device][0] - net_bytes_s[device][0]) / (end_time - start_time) / self.__unit_MiByte)
            net_transmit[
                '{}_transmit_bw'.format(device)
            ] = ((net_bytes_e[device][1] - net_bytes_s[device][1]) / (end_time - start_time) / self.__unit_MiByte)
        record.net_receive = net_receive
        record.net_transmit = net_transmit

    def __sample_gpu_metrics(self, record):
        """Method sampling the gpu metrics.

        Args:
            record (MonitorRecord): record instance to save the metrics.
        """
        device_count = dm.device_manager.get_device_count()
        for i in range(device_count):
            record.gpu_usage.append(dm.device_manager.get_device_utilization(i))
            record.gpu_temperature.append(dm.device_manager.get_device_temperature(i))
            record.gpu_power.append(dm.device_manager.get_device_power(i))
            record.gpu_power_limit.append(dm.device_manager.get_device_power_limit(i))
            mem_used, mem_total = dm.device_manager.get_device_memory(i)
            record.gpu_mem_used.append(mem_used)
            record.gpu_mem_total.append(mem_total)
            corrected_ecc, uncorrected_ecc = dm.device_manager.get_device_ecc_error(i)
            record.gpu_corrected_ecc.append(corrected_ecc)
            record.gpu_uncorrected_ecc.append(uncorrected_ecc)
            record.gpu_remap_info.append(dm.device_manager.get_device_row_remapped_info(i))

    def __get_total_cpu_ticks(self):
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
            logger.error('Failed to read total cpu ticks information - error message: {}'.format(str(e)))

        return None

    def __get_process_cpu_ticks(self):
        """Method to get the process cpu ticks.

        Return:
            The process cpu ticks, None means fail to get the data.
        """
        user_time = 0
        system_time = 0
        try:
            with open(self._cpu_file, 'r') as f:
                if self.__cgroup == 1:
                    for line in f:
                        items = line.split()
                        if items[0] == 'user':
                            user_time = int(items[1])
                        elif items[0] == 'system':
                            system_time = int(items[1])
                else:
                    for line in f:
                        items = line.split()
                        if items[0] == 'user_usec':
                            user_time = int(items[1]) / 10000
                        elif items[0] == 'system_usec':
                            system_time = int(items[1]) / 10000
            return user_time + system_time
        except BaseException as e:
            logger.error('Failed to read process cpu ticks information - error message: {}'.format(str(e)))

        return None

    def __get_network_bytes(self):
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
            logger.error('Failed to read network traffic information - error message: {}'.format(str(e)))

        return None
