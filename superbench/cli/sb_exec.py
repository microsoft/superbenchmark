#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench sb exec command."""

import hydra

from superbench.common.utils import logger


@hydra.main(config_path='../config', config_name='default')
def main(config):
    """The main entrypoint for sb-exec."""
    logger.info(config)
    # executor = SuperBenchExecutor(config)
    # executor.exec()


if __name__ == '__main__':
    main()
