# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for pattern config."""
from superbench.common.utils import logger
from superbench.common.utils import gen_topo_aware_config


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
    config = [','.join(map(str, list(range(n))))]
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
        list: the generated config list, each item in the list is a str like "0,1;2,3".
    """
    config = []
    if n <= 0:
        logger.error('n is not positive')
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


def gen_k_batch_config(scale, n):
    """Generate VM groups config with specified batch scale.

    Args:
        k (int): the scale of batch.
        n (int): the number of participants.

    Returns:
        list: the generated config list, each item in the list is a str like "0,1;2,3".
    """
    config = []
    if scale is None:
        logger.error('scale is not specified')
        return config
    if scale <= 0 or n <= 0:
        logger.error('scale or n is not positive')
        return config
    if scale > n:
        logger.error('scale large than n')
        return config

    group = []
    rem = n % scale
    for i in range(0, n - rem, scale):
        group.append(','.join(map(str, list(range(i, i + scale)))))
    config = [';'.join(group)]
    return config


def covert_config_to_host(config, hostx):
    """Convert config format to host node.

    Args:
        hosts (list): the list of VM hostnames read from hostfile.
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


def gen_pattern_host(hosts, args):
    """Generate traffic pattern config from specified mode.

    Args:
        hosts (list): the list of VM hostnames read from hostfile.
        args (dir): the arguments from config yaml.

    Returns:
        list: the host group from traffic pattern config.
    """
    config = []
    n = len(hosts)
    if args.pattern == 'all-nodes':
        config = gen_all_nodes_config(n)
    elif args.pattern == 'pair-wise':
        config = gen_pair_wise_config(n)
    elif args.pattern == 'k-batch':
        config = gen_k_batch_config(args.scale, n)
    elif args.pattern == 'topo-aware':
        config = gen_topo_aware_config(hosts, args.ibstat, args.ibnetdiscover, args.min_dist, args.max_dist)
    else:
        logger.error('Unsupported traffic pattern: {}'.format(args.pattern))
    host_group = covert_config_to_host(config, hosts)
    return host_group
