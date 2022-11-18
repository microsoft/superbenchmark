# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for traffic pattern config."""
from superbench.common.utils import logger


def gen_all_nodes_config(n):
    """Generate all nodes config.

    Args:
        n (int): the number of participants.

    Returns:
        list: the generated config list, each item in the list is a str like "0,1,2,3".
    """
    config = []
    if n <= 0:
        logger.error('n is not positive')
        return config
    config = [','.join(map(str, range(n)))]
    return config


def covert_config_to_host_group(config, hostx):
    """Convert config format to host node.

    Args:
        hostx (list): the list of hostnames read from hostfile.
        config (list): the traffic pattern config.

    Returns:
        list: the host group from traffic pattern config.
    """
    host_group = []
    for item in config:
        groups = item.strip().strip(';').split(';')
        host_list = []
        for group in groups:
            hosts = []
            for index in group.split(','):
                hosts.append(hostx[int(index)])
            host_list.append(hosts)
        host_group.append(host_list)
    return host_group


def gen_tarffic_pattern_host_group(host_list, args):
    """Generate traffic pattern config from specified mode.

    Args:
        host_list (list): the list of hostnames read from hostfile.
        args (dir): the arguments from config yaml.

    Returns:
        host_group (list): the host group from traffic pattern config.
    """
    config = []
    n = len(host_list)
    if args.pattern == 'all-nodes':
        config = gen_all_nodes_config(n)
    else:
        logger.error('Unsupported traffic pattern: {}'.format(args.pattern))
    host_group = covert_config_to_host_group(config, host_list)
    return host_group
