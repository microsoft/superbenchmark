# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from enum import Enum


class Platform(Enum):
    '''The Enum class representing different Platfrom.
    '''
    cpu = 0
    cuda = 1
    rocm = 2
    unknown = 3


def detect_platform():
    '''Detect the type of platform.

    Return:
        the type of platform, such as cuda, rocm.
    '''
    # NVIDIA GPU
    if os.path.exists('/dev/nvidiactl'):
        return Platform.cuda
    # AMD GPU
    elif os.path.exists('/dev/kfd'):
        return Platform.rocm
    # GPU device not mounted or driver not installed correctly
    else:
        return Platform.cpu
