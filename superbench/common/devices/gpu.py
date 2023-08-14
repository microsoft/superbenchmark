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
            if not list(Path('/dev/dri').glob('renderD*')):
                logger.warning('Cannot find AMD GPU device.')
            return 'amd'
        if list(Path(r'C:\Windows\System32').glob('*DriverStore/FileRepository/nv*.inf_amd64_*/nvapi64.dll')):
            return 'nvidia-graphics'
        if list(Path(r'C:\Windows\System32').glob('*DriverStore/FileRepository/u*.inf_amd64_*/*/aticfx64.dll')):
            return 'amd-graphics'
        return None

    @property
    def vendor(self):
        """Get the GPU vendor."""
        return self._vendor
