# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPU device module."""

from pathlib import Path

from superbench.common.utils import logger


class GPU():
    """GPU device helper class."""
    def __init__(self):
        """Initilize."""
        self._vendor = self.get_vendor()
        self._count = self.get_count()
        # TODO: check CUDA or ROCm availability accordingly

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
            if not list(Path('/dev/dri').glob('card*')):
                logger.warning('Cannot find AMD GPU device.')
            return 'amd'
        return None

    @property
    def vendor(self):
        """Get the GPU vendor."""
        return self._vendor

    def get_count(self):
        """Get GPU count.

        Returns:
            int: GPU count, nvidia or amd. 0 if no GPU found.
        """
        if 'nvidia' == self._vendor:
            return len(list(Path('/dev').glob('nvidia[0-9]*')))
        elif 'amd' == self._vendor:
            bdf_prefix = '[0-9a-fA-F]' * 4 + ':' + '[0-9a-fA-F]' * 2 + ':' + '[0-9a-fA-F]' * 2 + '.' + '[0-9a-fA-F]*'
            return len(list(Path('/sys/module/amdgpu/drivers/pci:amdgpu/').glob(bdf_prefix)))
        else:
            return 0

    @property
    def count(self):
        """Get the GPU count."""
        return self._count
