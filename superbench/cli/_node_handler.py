# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI node subgroup command handler."""

from superbench.tools import SystemInfo


def info_command_handler():
    """Get node hardware info.

    Returns:
        dict: node info.
    """
    try:
        info = SystemInfo().get_all()
    except Exception as ex:
        raise RuntimeError('Failed to get node info.') from ex
    return info
