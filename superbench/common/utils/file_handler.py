# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for file."""

from pathlib import Path
from datetime import datetime

from omegaconf import OmegaConf


def create_output_dir():
    """Create a new output directory.

    Generate a new output directory name based on current time and create it on filesystem.

    Returns:
        str: Output directory name.
    """
    output_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = Path('.', 'outputs', output_name).resolve()
    output_path.mkdir(mode=0o755, parents=True, exist_ok=True)
    return str(output_path)


def get_sb_config(config_file):
    """Read SuperBench config yaml.

    Read config file, use default config if None is provided.

    Args:
        config_file (str): config file path.

    Returns:
        OmegaConf: Config object, None if file does not exist.
    """
    default_config_file = Path(__file__).parent / '../../config/default.yaml'
    p = Path(config_file) if config_file else default_config_file
    if not p.is_file():
        return None
    return OmegaConf.load(str(p))
