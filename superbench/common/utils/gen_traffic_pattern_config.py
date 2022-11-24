# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for traffic pattern config."""
from superbench.common.utils import logger


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


def gen_tarffic_pattern_host_group(host_list, pattern):
    """Generate traffic pattern config from specified mode.

    Args:
        host_list (list): the list of hostnames read from hostfile.
        pattern (DictConfig): the mpi pattern dict.

    Returns:
        host_group (list): the host group from traffic pattern config.
    """
    config = []
    n = len(host_list)
    if pattern.name == 'all-nodes':
        config = gen_all_nodes_config(n)
    else:
        logger.error('Unsupported traffic pattern: {}'.format(pattern.name))
    host_group = __convert_config_to_host_group(config, host_list)
    return host_group
