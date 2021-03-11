# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for random_dataset module."""

import torch

from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


def test_torch_random_dataset():
    """Test TorchRandomDataset class."""
    shape = [32, 64]
    world_size = 1
    supported_types = [torch.float32, torch.float64, torch.int8, torch.int16, torch.int32, torch.int64]
    for dtype in supported_types:
        dataset = TorchRandomDataset(shape, world_size, dtype=dtype)
        assert (len(dataset) == 32)
        assert (len(dataset[0]) == 64)
        assert (dataset._data.dtype == dtype)

    world_size = 2
    for dtype in supported_types:
        dataset = TorchRandomDataset(shape, world_size, dtype=dtype)
        assert (len(dataset) == 64)
        assert (len(dataset[0]) == 64)
        assert (dataset._data.dtype == dtype)

    # Case for unsupported data type.
    dataset = TorchRandomDataset(shape, world_size, dtype=torch.float16)
    assert (len(dataset) == 0)
    assert (dataset._data is None)
