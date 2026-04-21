# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI node subgroup command handler."""

from pathlib import Path
import json

from knack.util import CLIError

from superbench.tools import SystemInfo
from superbench.common.utils import create_sb_output_dir
from superbench.common.utils.gpu_topology import (
    get_gpu_numa_affinity,
    get_gpu_numa_map,
)


def info_command_handler(output_dir=None):
    """Get node hardware info.

    Args:
        output_dir (str): Output directory.

    Returns:
        dict: node info.
    """
    try:
        info = SystemInfo().get_all()
        output_dir = create_sb_output_dir(output_dir)
        output_dir_path = Path(output_dir)
        with open(output_dir_path / 'sys_info.json', 'w') as f:
            json.dump(info, f)
    except Exception as ex:
        raise RuntimeError('Failed to get node info.') from ex
    return info


def topo_command_handler(get=None, gpu_id=None):
    """Get node topology information.

    Args:
        get (str): Topology field to get.
        gpu_id (int): GPU id.
    """
    if get == 'gpu-numa-map':
        print(json.dumps(get_gpu_numa_map()))
        return
    if get != 'gpu-numa-affinity':
        raise CLIError('Unsupported topology field: {}.'.format(get))
    if gpu_id is None:
        raise CLIError('--gpu-id is required for {}.'.format(get))

    print(get_gpu_numa_affinity(gpu_id))
