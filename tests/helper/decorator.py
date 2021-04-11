# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unittest decorator helpers."""

import os
import unittest

cuda_test = unittest.skipIf(os.environ.get('SB_TEST_CUDA', '1') == '0', 'Skip CUDA tests.')
rocm_test = unittest.skipIf(os.environ.get('SB_TEST_ROCM', '0') == '0', 'Skip ROCm tests.')

pytorch_test = unittest.skipIf(os.environ.get('SB_TEST_PYTORCH', '1') == '0', 'Skip PyTorch tests.')
