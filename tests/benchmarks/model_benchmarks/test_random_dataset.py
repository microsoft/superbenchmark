# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for random_dataset module."""

import torch

from superbench.benchmarks.model_benchmarks.random_dataset import TorchRandomDataset


def test_torch_random_dataset():
    """Test TorchRandomDataset class."""
    shape = [32, 64]
    world_size = 1
    dataset = TorchRandomDataset(shape, world_size, dtype=torch.float32)
    assert (len(dataset) == 32)
    assert (len(dataset[0]) == 64)
    assert (dataset._data.type() == 'torch.FloatTensor')

    world_size = 2
    dataset = TorchRandomDataset(shape, world_size, dtype=torch.int32)
    assert (len(dataset) == 64)
    assert (len(dataset[0]) == 64)
    assert (dataset._data.type() == 'torch.IntTensor')
