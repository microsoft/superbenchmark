#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench sb run command."""

import hydra

from superbench.common.utils import logger


@hydra.main(config_path='../config', config_name='default')
def main(config):
    """The main entrypoint for sb-run."""
    logger.info(config)
    # runner = SuperBenchRunner(config)
    # runner.run()


if __name__ == '__main__':
    main()
