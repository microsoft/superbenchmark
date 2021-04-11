# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for benchmark tests."""

import os


def setup_simulated_ddp_distributed_env():
    """Function to setup the simulated DDP distributed envionment variables."""
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'


def clean_simulated_ddp_distributed_env():
    """Function to clean up the simulated DDP distributed envionment variables."""
    os.environ.pop('WORLD_SIZE')
    os.environ.pop('RANK')
    os.environ.pop('LOCAL_RANK')
    os.environ.pop('MASTER_ADDR')
    os.environ.pop('MASTER_PORT')
