# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPU device module."""

from pathlib import Path

from superbench.benchmarks import Platform
from superbench.common.utils import logger


class GPU():
    """GPU device helper class."""
    def __init__(self):
        """Initilize."""
        self.vendor = self.get_vendor()
        # TODO: check CUDA or ROCm availability accordingly
        if self.vendor == 'nvidia':
            self.platform = Platform.CUDA
        elif self.vendor == 'amd':
            self.platform == Platform.ROCM
        else:
            self.platform = Platform.CPU

    def get_vendor(self):
        """Get GPU vendor.

        Returns:
            str: GPU vendor, nvidia or amd.
        """
        if Path('/dev/nvidiactl').is_char_device() and Path('/dev/nvidia-uvm').is_char_device():
            if not list(Path('/dev').glob('nvidia[0-9]*')):
                logger.warning('Cannot find NVIDIA GPU device.')
            return 'nvidia'
        if Path('/dev/kfd').is_char_device() and Path('/dev/dri').is_dir():
            if not list(Path('/dev/dri')).glob('card*'):
                logger.warning('Cannot find AMD GPU device.')
            return 'amd'
        return None
