# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for file."""

import itertools
from pathlib import Path
from datetime import datetime

import yaml
from omegaconf import OmegaConf

from superbench.common.utils import logger


def rotate_dir(target_dir):
    """Rotate directory if it is not empty.

    Args:
        target_dir (str): Target directory path.
    """
    try:
        if target_dir.is_dir() and any(target_dir.iterdir()):
            logger.warning('Directory %s is not empty.', str(target_dir))
            for i in itertools.count(start=1):
                backup_dir = target_dir.with_name(f'{target_dir.name}.{i}')
                if not backup_dir.is_dir():
                    target_dir.rename(backup_dir)
                    break
    except Exception:
        logger.exception('Failed to rotate directory %s.', str(target_dir))
        raise


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
    try:
        output_path.mkdir(mode=0o755, parents=True, exist_ok=True)
    except Exception:
        logger.exception('Failed to create directory %s.', str(output_path))
        raise
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
