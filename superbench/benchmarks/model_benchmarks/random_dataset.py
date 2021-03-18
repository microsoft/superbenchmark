# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module to define random Dataset."""

import torch
from torch.utils.data import Dataset

from superbench.common.utils import logger


class TorchRandomDataset(Dataset):
    """Dataset that can generate the input data randomly."""
    def __init__(self, shape, world_size, dtype=torch.float):
        """Constructor.

        Args:
            shape (List[int]): Shape of dataset.
            world_size (int): Number of workers.
            dtype (torch.dtype): Type of the elements.
        """
        self._len = 0
        self._data = None

        try:
            if dtype in [torch.float32, torch.float64]:
                self._data = torch.randn(*shape, dtype=dtype)
            elif dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                self._data = torch.randint(0, 128, tuple(shape), dtype=dtype)
            else:
                logger.error('Unsupported precision for RandomDataset - data type: {}.'.format(dtype))
                return
        except BaseException as e:
            logger.error(
                'Generate random dataset failed - data type: {}, shape: {}, message: {}.'.format(dtype, shape, str(e))
            )
            return

        self._len = shape[0] * world_size
        self._world_size = world_size

    def __getitem__(self, index):
        """Get the element according to index.

        Args:
            index (int): Position index.

        Return:
            Element in dataset.
        """
        return self._data[int(index / self._world_size)]

    def __len__(self):
        """Get the count of elements.

        Return:
            The count of elements.
        """
        return self._len
