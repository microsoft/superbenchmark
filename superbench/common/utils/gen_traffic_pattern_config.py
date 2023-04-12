# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for traffic pattern config."""
import re
from pathlib import Path

from superbench.common.utils import logger
from superbench.common.utils import gen_topo_aware_config


def gen_all_nodes_config(n):
    """Generate all nodes config.

    Args:
        n (int): the number of participants.

    Returns:
        config (list): the generated config list, each item in the list is a str like "0,1,2,3".
    """
    config = []
    if n <= 0:
        logger.warning('n is not positive')
        return config
    config = [','.join(map(str, range(n)))]
    return config


def gen_pair_wise_config(n):
    """Generate pair-wised VM pairs config.

    One-to-one means that each participant plays every other participant once.
    The algorithm refers circle method of Round-robin tournament in
    https://en.wikipedia.org/wiki/Round-robin_tournament.
    if n is even, there are a total of n-1 rounds, with n/2 pair of 2 unique participants in each round.
    If n is odd, there will be n rounds, each with n-1/2 pairs, and one participant rotating empty in that round.
    In each round, pair up two by two from the beginning to the middle as (begin, end),(begin+1,end-1)...
    Then, all the participants except the beginning shift left one position, and repeat the previous step.

    Args:
        n (int): the number of participants.

    Returns:
        config (list): the generated config list, each item in the list is a str like "0,1;2,3".
    """
    config = []
    if n <= 0:
        logger.warning('n is not positive')
        return config
    candidates = list(range(n))
    # Add a fake participant if n is odd
    if n % 2 == 1:
        candidates.append(-1)
    count = len(candidates)
    non_moving = [candidates[0]]
    for _ in range(count - 1):
        pairs = [
            '{},{}'.format(candidates[i], candidates[count - i - 1]) for i in range(0, count // 2)
            if candidates[i] != -1 and candidates[count - i - 1] != -1
        ]
        row = ';'.join(pairs)
        config.append(row)
        robin = candidates[2:] + candidates[1:2]
        candidates = non_moving + robin
    return config


def gen_k_batch_config(n, batch):
    """Generate VM groups config with specified batch scale.

    Args:
        n (int): the number of participants.
        batch (int): the scale of batch.

    Returns:
        config (list): the generated config list, each item in the list is a str like "0,1;2,3".
    """
    config = []
    if batch is None:
        logger.warning('scale is not specified')
        return config
    if batch <= 0 or n <= 0:
        logger.warning('scale or n is not positive')
        return config
    if batch > n:
        logger.warning('scale large than n')
        return config

    group = []
    rem = n % batch
    for i in range(0, n - rem, batch):
        group.append(','.join(map(str, list(range(i, i + batch)))))
    config = [';'.join(group)]
    return config


def __convert_config_to_host_group(config, host_list):
    """Convert config format to host node.

    Args:
        host_list (list): the list of hostnames read from hostfile.
        config (list): the traffic pattern config.

    Returns:
        host_groups (list): the host groups converted from traffic pattern config.
    """
    host_groups = []
    for item in config:
        groups = item.strip().strip(';').split(';')
        host_group = []
        for group in groups:
            hosts = []
            for index in group.split(','):
                hosts.append(host_list[int(index)])
            host_group.append(hosts)
        host_groups.append(host_group)
    return host_groups


def gen_ibstat(ansible_config, ibstat_path):    # pragma: no cover
    """Generate the ibstat file in specified path.

    Args:
        ansible_config (DictConfig): Ansible config object.
        ibstat_path (str): the expected path of ibstat file.

    Returns:
        ibstat_path (str): the generated path of ibstat file.
    """
    from superbench.runner import AnsibleClient
    ibstat_list = []
    stdout_regex = re.compile(r'\x1b(\[.*?[@-~]|\].*?(\x07|\x1b\\))')
    ansible_client = AnsibleClient(ansible_config)
    cmd = 'cat /sys/class/infiniband/*/sys_image_guid | tr -d :'

    # callback function to collect and parse ibstat
    def _ibstat_parser(artifact_dir):
        stdout_path = Path(artifact_dir) / 'stdout'
        with stdout_path.open(mode='r') as raw_outputs:
            for raw_output in raw_outputs:
                output = stdout_regex.sub('', raw_output).strip()
                if ' | CHANGED | rc=0 >>' in output:
                    output = 'VM_hostname ' + output.replace(' | CHANGED | rc=0 >>', '')
                ibstat_list.append(output)

    config = ansible_client.get_shell_config(cmd)
    config['artifacts_handler'] = _ibstat_parser
    rc = ansible_client.run(config)
    if rc != 0:
        logger.error('Failed to gather ibstat with config: {}'.format(config))
    with Path(ibstat_path).open(mode='w') as f:
        for ibstat in ibstat_list:
            f.write(ibstat + '\n')
    return ibstat_path


def gen_traffic_pattern_host_groups(host_list, pattern, mpi_pattern_path, benchmark_name):
    """Generate host group from specified traffic pattern and write in specified path.

    Args:
        host_list (list): the list of hostnames read from hostfile.
        pattern (DictConfig): the mpi pattern dict.
        mpi_pattern_path (str): the path of mpi pattern config file.
        benchmark_name (str): the name of benchmark.

    Returns:
        host_groups (list): the host groups generated from traffic pattern.
    """
    config = []
    n = len(host_list)
    if pattern.type == 'all-nodes':
        config = gen_all_nodes_config(n)
    elif pattern.type == 'pair-wise':
        config = gen_pair_wise_config(n)
    elif pattern.type == 'k-batch':
        config = gen_k_batch_config(n, pattern.batch)
    elif pattern.type == 'topo-aware':
        config = gen_topo_aware_config(
            host_list, pattern.ibstat, pattern.ibnetdiscover, pattern.min_dist, pattern.max_dist
        )
    else:
        logger.error('Unsupported traffic pattern: {}'.format(pattern.type))
    host_groups = __convert_config_to_host_group(config, host_list)
    # write traffic pattern host groups to specified path
    with open(mpi_pattern_path, 'a') as f:
        f.write('benchmark_name: {} pattern_type: {}'.format(benchmark_name, pattern.type) + '\n')
        for host_group in host_groups:
            row = []
            for host_list in host_group:
                group = ','.join(host_list)
                row.append(group)
            group = ';'.join(row)
            f.write(group + '\n')
        f.write('\n')
    return host_groups
