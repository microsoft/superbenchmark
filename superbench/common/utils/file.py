# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for file."""

import yaml
from pathlib import Path
from datetime import datetime


def new_output_dir():
    """Generate a new output directory.

    Generate a new output directory name based on current time and create it on filesystem.

    Returns:
        str: Output directory name.
    """
    output_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = Path('.', 'outputs', output_name)
    output_path.mkdir(mode=0o755, parents=True, exist_ok=True)
    return str(output_path)


def get_config(config_file):
    """Read SuperBench config yaml.

    Read config file, use default config if None is provided.

    Args:
        config_file (str): config file path.

    Returns:
        dict: Config object.
    """
    here = Path(__file__).parent.resolve()
    p = Path(config_file) if config_file else here / '../../config/default.yaml'
    with p.open() as f:
        config = yaml.safe_load(f)
    return config
