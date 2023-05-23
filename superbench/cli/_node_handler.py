# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI node subgroup command handler."""

import json
from pathlib import Path

from superbench.tools import SystemInfo
from superbench.common.utils import create_sb_output_dir


def info_command_handler(output_dir=None):
    """Get node hardware info.

    Args:
        output_dir (str): Output directory.
    """
    try:
        output_dir = create_sb_output_dir(output_dir)
        info = SystemInfo().get_all()
        output_dir_path = Path(output_dir)
        with open(output_dir_path / 'sys_info.json', 'w') as f:
            json.dump(info, f)
    except Exception as ex:
        raise RuntimeError('Failed to get node info.') from ex
