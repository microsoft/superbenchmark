# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for file."""

from pathlib import Path
from datetime import datetime

import yaml
from omegaconf import OmegaConf


def create_sb_output_dir(output_dir=None):
    """Create output directory.

    Create output directory on filesystem, generate a new name based on current time if not provided.

    Args:
        output_dir (str): Output directory. Defaults to None.

    Returns:
        str: Given or generated output directory.
    """
    if not output_dir:
        output_dir = str(Path('outputs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(mode=0o755, parents=True, exist_ok=True)
    return output_dir


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
    with p.open() as fp:
        return OmegaConf.create(yaml.load(fp, Loader=yaml.SafeLoader))
