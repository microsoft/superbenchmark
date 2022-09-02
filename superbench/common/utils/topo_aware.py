# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Topology Aware Utilities."""

import re
import os
from pathlib import Path

import networkx as nx

from superbench.common.utils import logger


class quick_regexp(object):
    """Quick regular expression class."""
    def __init__(self):
        """Constructor."""
        self.groups = None
        self.matched = False

    def search(self, pattern, string, flags=0):
        """Search with group function."""
        match = re.search(pattern, string, flags)
        if match:
            self.matched = True
            if match.groups():
                self.groups = re.search(pattern, string, flags).groups()
            else:
                self.groups = True
        else:
            self.matched = False
            self.groups = None

        return self.matched


def gen_ibstat_file(host_list, ibstat_file):
    """Generate ibstat file in each node with specified path.

    Args:
        host_list (list): list of VM read from hostfile.
        ibstat_file (str): path of ibstat output.
    """
    try:
        pssh_cmd = "pssh -i -t 5 -H '{}' ".format(' '.join(host_list))
        cmd = "'cat /sys/class/infiniband/*/sys_image_guid | tr -d :' | sed 's/^.*[[[:upper:]]\{7\}]/VM_hostname/g'"
        output = os.popen(pssh_cmd + cmd).read()
        if 'error code' in output:
            logger.error('Failed to fetch guid info with specified hostlist')
        # Generate ibstat file
        ibstate_file_path = Path(ibstat_file)
        with ibstate_file_path.open(mode='w') as f:
            f.writelines(output)
    except BaseException as e:
        logger.error('Failed to generate ibstate file, message: {}.'.format(str(e)))


def gen_topo_aware_config(host_list, ibstat_file, ibnetdiscover_file, min_dist, max_dist):    # noqa: C901
    """Generate topology aware config list in specified distance range.

    Args:
        host_list (list): list of VM read from hostfile.
        ibstat_file (str): path of ibstat output.
        ibnetdiscover_file (str): path of ibnetdiscover output.
        min_dist (int): minimum distance of VM pair.
        max_dist (int): maximum distance of VM pair.

    Returns:
        list: the generated config list, each item in the list is a str
        like "0,1;2,3", which represents all VM pairs with a fixed
        topology distance (#hops).
    """
    config = []
    # Check validity of input parameters
    if not ibnetdiscover_file:
        logger.error('ibnetdiscover file is not specified.')
        return config

    if not ibstat_file:
        ibstat_file = os.path.join(os.environ.get('SB_WORKSPACE', '.'), 'ib_traffic_topo_aware_ibstat.txt')
        gen_ibstat_file(host_list, ibstat_file)

    if not Path(ibstat_file).exists():
        logger.error('ibstat file does not exist.')
        return config

    if min_dist > max_dist:
        logger.error('Specified minimum distane ({}) is larger than maximum distance ({}).'.format(min_dist, max_dist))
        return config

    # Index each hostname in hostfile
    host_idx = dict()
    idx = 0
    for h in host_list:
        host_idx[h.strip()] = idx
        idx += 1

    sysimgguid_to_vmhost = dict()
    phyhost_to_sysimgguid = dict()
    topology = dict()
    topologyX = dict()
    hosts = list()
    current_dev = ''

    # Read ibstat output, store mapping from sysimgguid to vmhost for each HCA
    try:
        with open(ibstat_file, mode='r', buffering=1) as f:
            for line in f:
                line = line.strip()
                isinstance(line, str)
                if line:
                    r = quick_regexp()
                    if r.search(r'^(VM_hostname)\s+(.+)', line):
                        vmhost = r.groups[1]
                    elif r.search(r'^(?!0{16})([a-z0-9]{16})$', line):
                        sysimgguid = r.groups[0]
                        sysimgguid_to_vmhost[sysimgguid] = vmhost
    except BaseException as e:
        logger.error('Failed to read ibstate file, message: {}.'.format(str(e)))
        return config

    # Read ibnetdiscover output to
    # store the information of each device (Swith/HCA)
    # collect all physical hosts that are associated with HCA
    # store mapping from physical hostname to sysimgguid for each HCA
    try:
        with open(ibnetdiscover_file, mode='r', buffering=1) as f:
            for line in f:
                line = line.strip()
                isinstance(line, str)
                if line:
                    r = quick_regexp()
                    # Read the device (Switch/HCA)'s port, GUID, and metadata
                    if r.search(r'^(\w+)\s+(\d+)\s+\"(.+?)\"\s+#\s+\"(.+?)\"', line):
                        current_dev = r.groups[2]
                        topology[current_dev] = dict()
                        topology[current_dev]['number_of_ports'] = int(r.groups[1])
                        topology[current_dev]['sysimgguid'] = r.groups[2].split('-')[1]
                        topology[current_dev]['metadata'] = r.groups[3]
                        metadata = topology[current_dev]['metadata']

                        if r.groups[0] == 'Switch':
                            topology[current_dev]['node_type'] = 'switch'
                            topologyX[metadata] = dict()
                            topologyX[metadata]['node_type'] = 'switch'
                        else:
                            topology[current_dev]['node_type'] = 'hca'
                            if topology[current_dev]['metadata'] == 'Mellanox Technologies Aggregation Node':
                                topology[current_dev]['hca_type'] = 'AN'
                            elif (topology[current_dev]['metadata'].find('ufm') != -1):
                                topology[current_dev]['hca_type'] = 'UFM'
                            else:
                                topology[current_dev]['hca_type'] = 'HCA'
                                hostname = metadata.split(' ')[0]
                                topology[current_dev]['hostname'] = hostname
                                curr_sysimgguid = topology[current_dev]['sysimgguid']
                                # curr_sysimgguid in sysimgguid_to_vmhost is to check
                                # if the physical host associated with current device has VM running on it.
                                # If not, no need to include this physical host for distance calculation later.
                                if (
                                    not (hostname in topologyX) and hostname != 'MT4123'
                                    and curr_sysimgguid in sysimgguid_to_vmhost
                                ):
                                    topologyX[hostname] = dict()
                                    topologyX[hostname]['node_type'] = 'host'
                                    hosts.append(hostname)
                                    phyhost_to_sysimgguid[hostname] = curr_sysimgguid

                    # Read the port connection lines under each device (switch/hca)
                    if r.search(r'^\[(\d+)\].*?\"(.+?)\"\[(\d+)\]', line):
                        local_port = int(r.groups[0])
                        connected_to_remote_host = r.groups[1]
                        connected_to_remote_port = int(r.groups[2])
                        topology[current_dev][local_port] = {connected_to_remote_host: connected_to_remote_port}
    except BaseException as e:
        logger.error('Failed to read ibnetdiscover file, message: {}.'.format(str(e)))
        return config

    # Build a graph across physical hosts and switch nodes
    Gnx = nx.Graph()
    Gnx.add_nodes_from(topologyX)
    for dev in topology:
        numPorts = topology[dev]['number_of_ports']
        if numPorts > 0:
            for port in range(1, numPorts + 1):
                if port in topology[dev].keys():
                    remote_host = list(topology[dev][port].keys())[0]
                    if topology[dev]['node_type'] == 'hca':
                        if topology[dev]['hca_type'] == 'AN' or topology[dev]['hca_type'] == 'UFM':
                            continue
                        src = topology[dev]['hostname']
                    else:
                        src = topology[dev]['metadata']

                    if remote_host in topology:
                        if topology[remote_host]['node_type'] == 'hca':
                            if topology[remote_host]['hca_type'] == 'AN' or topology[remote_host]['hca_type'] == 'UFM':
                                continue
                            dest = topology[remote_host]['hostname']
                        else:
                            dest = topology[remote_host]['metadata']

                        Gnx.add_edge(src, dest)

    all_paths_len_obj = nx.all_pairs_shortest_path_length(Gnx)
    all_paths_len = dict(all_paths_len_obj)

    # Generate VM pairs with different topology distance
    for nodeDistance in range(min_dist, max_dist + 1):
        visited = dict()
        hostpairs = {}
        row = []
        for hostx in hosts:
            if hostx in visited:
                continue
            visited[hostx] = 1
            for hosty in hosts:
                if hosty in visited:
                    continue
                if all_paths_len[hostx][hosty] == nodeDistance:
                    hostpairs[hostx] = hosty
                    visited[hosty] = 1
                    break
        for host in hostpairs:
            vma = sysimgguid_to_vmhost[phyhost_to_sysimgguid[host]]
            vmb = sysimgguid_to_vmhost[phyhost_to_sysimgguid[hostpairs[host]]]
            idx_pair = '{},{}'.format(host_idx[vma], host_idx[vmb])
            row.append(idx_pair)
        if row:
            row = ';'.join(row)
            config.append(row)

    return config
