# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unittest decorator helpers."""

import os
import unittest
import functools
from pathlib import Path

cuda_test = unittest.skipIf(os.environ.get('SB_TEST_CUDA', '1') == '0', 'Skip CUDA tests.')
rocm_test = unittest.skipIf(os.environ.get('SB_TEST_ROCM', '0') == '0', 'Skip ROCm tests.')

pytorch_test = unittest.skipIf(os.environ.get('SB_TEST_PYTORCH', '1') == '0', 'Skip PyTorch tests.')
directx_test = unittest.skipIf(os.environ.get('SB_TEST_DIRECTX', '0') == '0', 'Skip DirectX tests.')


def load_data(filepath):
    """Decorator to load data file.

    Args:
        filepath (str): Data file path, e.g., tests/data/output.log.

    Returns:
        func: decorated function, data variable is assigned to last argument.
    """
    with Path(filepath).open() as fp:
        data = fp.read()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, data, **kwargs)

        return wrapper

    return decorator
