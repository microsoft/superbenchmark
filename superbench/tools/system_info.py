# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Generate system config."""

import json
import os
import subprocess
from pathlib import Path

import xmltodict

from superbench.common.utils import logger


class SystemInfo():    # pragma: no cover
    """Systsem info class."""
    def _run_cmd(self, command):
        """Run the command and return the stdout string.

        Args:
            command (string): the command to run in terminal.

        Returns:
            string: the stdout string of the command.
        """
        output = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            check=False,
            universal_newlines=True,
            timeout=300
        )
        return output.stdout

    def __count_prefix_indent(self, content, symbol='\t'):
        r"""Count the number of a specific symbol in the content.

        Args:
            content (string): the content for counting the indent.
            symbol (str, optional): the symbol of the indent. Defaults to '\t'.

        Returns:
            int: the indent count of the symbol in the beginning of the content.
        """
        count = 0
        for char in content:
            if char == symbol:
                count += 1
            else:
                break
        return count

    def _parse_key_value_lines(self, lines, required_keywords=None, omitted_values=None, symbol=':'):    # noqa: C901
        """Parse the lines like "key:value" and convert them to dict.

        if required_keywords is None, include all line. Otherwise,
        only include the line containing one of the keyword in required_keywords.
        If omitted_values is None, accept any value in dict,
        otherwise drop the item whose value in omitted_values.

        Args:
            lines (list): the lines to parse.
            required_keywords (list, optional): list of select keys. Defaults to None.
            omitted_values (list, optional): list of omitted values. Defaults to None.

        Returns:
            dict: the result in dict.
        """
        dict = {}
        key = ''
        value = ''
        i = 0
        length = len(lines)
        while i < length:
            line = lines[i]
            is_selected = True
            if required_keywords is not None:
                is_selected = False
                for key in required_keywords:
                    if key in line:
                        is_selected = True
            if not is_selected:
                i += 1
                continue
            # process with indent recursively
            indent = self.__count_prefix_indent(lines[i])
            if i + 1 < length and self.__count_prefix_indent(lines[i + 1]) > indent:
                key = lines[i].strip().strip('\t')
                next_indent_index = i + 1
                while next_indent_index < length and self.__count_prefix_indent(lines[next_indent_index]) > indent:
                    next_indent_index += 1

                value = self._parse_key_value_lines(lines[i + 1:next_indent_index])
                i = next_indent_index - 1
            # split line by symbol
            elif symbol in line:
                symbol_index = line.index(symbol)
                line = [line[:symbol_index], line[symbol_index + 1:]]
                key = line[0].strip().strip('\t')
                if len(line) > 1:
                    value = line[1].strip().strip('\t')
                else:
                    value = ''

            is_omit = False
            if omitted_values is not None:
                for omit in omitted_values:
                    if omit in value.lower():
                        is_omit = True
            if not is_omit:
                # save key,value into the dict and merge same key
                if key not in dict:
                    dict[key] = value
                elif dict[key] is not value:
                    if not isinstance(dict[key], list):
                        dict[key] = [dict[key]]
                    if value not in dict[key]:
                        dict[key].append(value)
            i += 1
        return dict

    def _parse_table_lines(self, lines, key):
        """Parse lines like a table and extract the colomns whose table index are the same as key to list of dict.

        Args:
            lines (list): the lines to parse.
            key ([type]): A subset of the the table index.

        Returns:
            list: the result in list of dict.
        """
        index = []
        list = []
        valid = False
        for line in lines:
            line = line.split()
            if key[0] in line:
                for i in range(len(key)):
                    index.append(line.index(key[i]))
                valid = True
                continue
            if valid:
                dict = {}
                for i in range(len(key)):
                    if index[i] < len(line):
                        dict[key[i]] = line[index[i]]
                list.append(dict)
        return list

    def get_cpu(self):
        """Get CPU info.

        Returns:
            dict: cpu info dict.
        """
        lscpu_dict = {}
        try:
            # get general cpu information from lscpu
            lscpu = self._run_cmd('lscpu').splitlines()
            # get distinct max_speed and current_speed of cpus from dmidecode
            speed = self._run_cmd(r'dmidecode -t processor | grep "Speed"').splitlines()
            lscpu_dict = self._parse_key_value_lines(lscpu)
            lscpu_dict.update(self._parse_key_value_lines(speed))
        except Exception:
            logger.exception('Error: get CPU info failed')
        return lscpu_dict

    def get_system(self):
        """Get system info.

        Returns:
            dict: system info dict.
        """
        system_dict = {}
        try:
            lsmod = self._run_cmd('lsmod').splitlines()
            lsmod = self._parse_table_lines(lsmod, key=['Module', 'Size', 'Used', 'by'])
            sysctl = self._run_cmd('sysctl -a').splitlines()
            sysctl = self._parse_key_value_lines(sysctl, None, None, '=')
            system_dict['system_manufacturer'] = self._run_cmd('dmidecode -s system-manufacturer').strip()
            system_dict['system_product'] = self._run_cmd('dmidecode -s system-product-name').strip()
            system_dict['os'] = self._run_cmd('cat /proc/version').strip()
            system_dict['uname'] = self._run_cmd('uname -a').strip()
            system_dict['docker'] = self.get_docker_version()
            system_dict['kernel_parameters'] = sysctl
            system_dict['kernel_modules'] = lsmod
            system_dict['dmidecode'] = self._run_cmd('dmidecode').strip()
            if system_dict['system_product'] == 'Virtual Machine':
                lsvmbus = self._run_cmd('lsvmbus').splitlines()
                lsvmbus = self._parse_key_value_lines(lsvmbus)
                system_dict['vmbus'] = lsvmbus
        except Exception:
            logger.exception('Error: get system info failed')
        return system_dict

    def get_docker_version(self):
        """Get docker version info.

        Returns:
            dict: docker version info dict.
        """
        docker_version_dict = {}
        try:
            docker_version = self._run_cmd('docker version')
            lines = docker_version.splitlines()

            key = ''
            for line in lines:
                if 'Client' in line:
                    key = 'docker_client_version'
                elif 'Server' in line:
                    key = 'docker_daemon_version'
                elif 'Version' in line and key not in docker_version_dict:
                    docker_version_dict[key] = line.split(':')[1].strip().strip('\t')
        except Exception:
            logger.exception('Error: get docker info failed')
        return docker_version_dict

    def get_memory(self):
        """Get memory info.

        Returns:
            dict: memory info dict.
        """
        memory_dict = {}
        try:
            lsmem = self._run_cmd('lsmem')
            lsmem = lsmem.splitlines()
            lsmem = self._parse_key_value_lines(lsmem)
            memory_dict['block_size'] = lsmem.get('Memory block size', '')
            memory_dict['total_capacity'] = lsmem.get('Total online memory', '')
            dmidecode_memory = self._run_cmd('dmidecode --type memory')
            dmidecode_memory = dmidecode_memory.splitlines()
            model = self._parse_key_value_lines(
                dmidecode_memory, ['Manufacturer', 'Part Number', 'Type', 'Speed', 'Number Of Devices'],
                omitted_values=['other', 'unknown']
            )
            memory_dict['channels'] = model.get('Number Of Devices', '')
            memory_dict['type'] = model.get('Type', '')
            memory_dict['clock_frequency'] = model.get('Speed', '')
            memory_dict['model'] = model.get('Manufacturer', [''])[0] + ' ' + model.get('Part Number', [''])[0]
        except Exception:
            logger.exception('Error: get memory info failed')
        return memory_dict

    def get_gpu_nvidia(self):
        """Get nvidia gpu info.

        Returns:
            dict: nvidia gpu info dict.
        """
        gpu_dict = {}
        gpu_query = self._run_cmd('nvidia-smi -q -x')
        gpu_query = xmltodict.parse(gpu_query).get('nvidia_smi_log', '')
        gpu_dict['gpu_count'] = gpu_query.get('attached_gpus', '')
        gpu_dict['nvidia_info'] = gpu_query
        gpu_dict['topo'] = self._run_cmd('nvidia-smi topo -m')
        gpu_dict['nvidia-container-runtime_version'] = self._run_cmd('nvidia-container-runtime -v').strip()
        gpu_dict['nvidia-fabricmanager_version'] = self._run_cmd('nv-fabricmanager --version').strip()
        gpu_dict['nv_peer_mem_version'] = self._run_cmd(
            'dpkg -l | grep \'nvidia-peer-memory \' | awk \'$2=="nvidia-peer-memory" {print $3}\''
        ).strip()

        return gpu_dict

    def get_gpu_amd(self):
        """Get amd gpu info.

        Returns:
            dict: amd gpu info dict.
        """
        gpu_dict = {}
        gpu_query = self._run_cmd('rocm-smi -a --json')
        gpu_query = json.loads(gpu_query)
        gpu_per_node = list(filter(lambda x: 'card' in x, gpu_query.keys()))
        gpu_dict['gpu_count'] = len(gpu_per_node)
        gpu_mem_info = self._run_cmd('rocm-smi --showmeminfo vram --json')
        gpu_mem_info = json.loads(gpu_mem_info)
        for card in gpu_per_node:
            gpu_query[card].update(gpu_mem_info.get(card))
        gpu_dict['rocm_info'] = gpu_query
        gpu_dict['topo'] = self._run_cmd('rocm-smi --showtopo')

        return gpu_dict

    def get_gpu(self):
        """Get gpu info and identify gpu type(nvidia/amd).

        Returns:
            dict: gpu info dict.
        """
        try:
            if Path('/dev/nvidiactl').is_char_device() and Path('/dev/nvidia-uvm').is_char_device():
                return self.get_gpu_nvidia()
            if Path('/dev/kfd').is_char_device() and Path('/dev/dri').is_dir():
                return self.get_gpu_amd()
        except Exception:
            logger.exception('Error: get gpu info failed')
        print('Warning: no gpu detected')
        return {}

    def get_pcie(self):
        """Get pcie info dict.

        Returns:
            dict: pcie info dict.
        """
        pcie_dict = {}
        try:
            pcie_dict['pcie_topo'] = self._run_cmd('lspci -t -vvv')
            pcie_dict['pcie_info'] = self._run_cmd('lspci -vvv')
        except Exception:
            logger.exception('Error: get pcie info failed')
        return pcie_dict

    def get_storage(self):    # noqa: C901
        """Get storage info dict, including file system info, blocl device info and their mapping.

        Returns:
            dict: storage info dict.
        """
        storage_dict = {}
        try:
            fs_info = self._run_cmd("df -Th | grep -v \'^/dev/loop\'").splitlines()
            fs_list = self._parse_table_lines(fs_info, key=['Filesystem', 'Type', 'Size', 'Avail', 'Mounted'])
            for fs in fs_list:
                fs_device = fs.get('Filesystem', 'UNKNOWN')
                if fs_device.startswith('/dev'):
                    fs['Block_size'] = self._run_cmd('blockdev --getbsz {}'.format(fs_device)).strip()
                    fs['4k_alignment'] = ''
                    partition_ids = self._run_cmd(
                        'yes Cancel | parted {} print | grep -oE "^[[:blank:]]*[0-9]+"'.format(fs_device)
                    ).splitlines()
                    for id in partition_ids:
                        fs['4k_alignment'] += self._run_cmd(
                            'yes Cancel | parted {} align-check opt {}'.format(fs_device, id)
                        ).strip()
            storage_dict['file_system'] = fs_list
        except Exception:
            logger.exception('Error: get file system info failed')

        try:
            disk_info = self._run_cmd("lsblk -e 7 -o NAME,ROTA,SIZE,MODEL | grep -v \'^/dev/loop\'").splitlines()
            disk_list = self._parse_table_lines(disk_info, key=['NAME', 'ROTA', 'SIZE', 'MODEL'])
            for disk in disk_list:
                block_device = disk.get('NAME', 'UNKNOWN').strip('\u251c\u2500').strip('\u2514\u2500')
                disk['NAME'] = block_device
                disk['Rotational'] = disk.pop('ROTA')
                disk['Block_size'] = self._run_cmd('fdisk -l -u /dev/{} | grep "Sector size"'.format(block_device)
                                                   ).strip()
                if 'nvme' in block_device:
                    nvme_info = self._run_cmd('nvme list | grep {}'.format(block_device)).strip().split()
                    if len(nvme_info) >= 15:
                        disk['Nvme_usage'] = nvme_info[-11] + nvme_info[-10]
            storage_dict['block_device'] = disk_list
            storage_dict['mapping_bwtween_filesystem_and_blockdevice'] = self._run_cmd('mount')
        except Exception:
            logger.exception('Error: get block device info failed')

        return storage_dict

    def get_ib(self):
        """Get available IB devices info.

        Return:
            list: list of available IB device info dict.
        """
        ib_dict = {}
        try:
            ibstat = self._run_cmd('ibstat').splitlines()
            ib_dict['ib_device_status'] = self._parse_key_value_lines(ibstat)
            ibv_devinfo = self._run_cmd('ibv_devinfo -v').splitlines()
            for i in range(len(ibv_devinfo) - 1, -1, -1):
                if ':' not in ibv_devinfo[i]:
                    ibv_devinfo[i - 1] = ibv_devinfo[i - 1] + ',' + ibv_devinfo[i].strip('\t')
                    ibv_devinfo.remove(ibv_devinfo[i])
            ib_dict['ib_device_info'] = self._parse_key_value_lines(ibv_devinfo)
        except Exception:
            logger.exception('Error: get ib info failed')
        return ib_dict

    def get_nic(self):
        """Get nic info.

        Returns:
            list: list of available nic info dict.
        """
        nic_list = []
        try:
            lsnic_xml = self._run_cmd('lshw -c network -xml')
            lsnic_list = xmltodict.parse(lsnic_xml).get('list', {}).get('node', [])
            if not isinstance(lsnic_list, list):
                lsnic_list = [lsnic_list]
            lsnic_list = list(filter(lambda x: 'logicalname' in x, lsnic_list))

            for nic in lsnic_list:
                nic_info = {}
                try:
                    nic_info['logical_name'] = nic['logicalname']
                    nic_info['disabled'] = nic.get('@disabled', False)
                    nic_info['model'] = nic.get('vendor', '') + ' ' + nic.get('product', '')
                    nic_info['description'] = nic.get('description', '')
                    configuration = nic.get('configuration', {}).get('setting')
                    configuration_dict = {}
                    for config in configuration:
                        configuration_dict[config['@id']] = config.get('@value', '')
                    if configuration_dict:
                        nic_info['driver'] = configuration_dict.get('driver', '') + ' ' + configuration_dict.get(
                            'driverversion', ''
                        )
                        nic_info['firmware'] = configuration_dict.get('firmware', '')
                    speed = self._run_cmd('cat /sys/class/net/{}/speed'.format(nic_info['logical_name'])).strip()
                    if speed.isdigit():
                        nic_info['speed'] = str(int(speed) / 1000) + ' Gbit/s'
                except Exception:
                    logger.exception('Error: get nic device {} info failed')

                nic_list.append(nic_info)
        except Exception:
            logger.exception('Error: get nic info failed')

    def get_network(self):
        """Get network info, including nic info, ib info and ofed version.

        Returns:
            dict: dict of network info.
        """
        network_dict = {}
        try:
            network_dict['nic'] = self.get_nic()
            network_dict['ib'] = self.get_ib()
            ofed_version = self._run_cmd('ofed_info  -s').strip()
            network_dict['ofed_version'] = ofed_version
        except Exception:
            logger.exception('Error: get network info failed')
        return network_dict

    def get_all(self):
        """Get all system info and save them to file in json format."""
        sum_dict = {}
        if os.geteuid() != 0:
            logger.error('You need to be as a root user to run this tool.')
            return sum_dict
        sum_dict['System'] = self.get_system()
        sum_dict['CPU'] = self.get_cpu()
        sum_dict['Memory'] = self.get_memory()
        sum_dict['Storage'] = self.get_storage()
        sum_dict['Network'] = self.get_network()
        sum_dict['PCIe'] = self.get_pcie()
        sum_dict['Accelerator'] = self.get_gpu()
        return sum_dict
