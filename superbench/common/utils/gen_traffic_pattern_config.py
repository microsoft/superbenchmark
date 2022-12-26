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


def gen_traffic_pattern_host_group(host_list, pattern):
    """Generate host group from specified traffic pattern.

    Args:
        host_list (list): the list of hostnames read from hostfile.
        pattern (DictConfig): the mpi pattern dict.

    Returns:
        host_group (list): the host group generated from traffic pattern.
    """
    config = []
    n = len(host_list)
    if pattern.name == 'all-nodes':
        config = gen_all_nodes_config(n)
    elif pattern.name == 'pair-wise':
        config = gen_pair_wise_config(n)
    else:
        logger.error('Unsupported traffic pattern: {}'.format(pattern.name))
    host_group = __convert_config_to_host_group(config, host_list)
    return host_group
