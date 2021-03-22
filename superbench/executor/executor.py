# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Executor."""

from pathlib import Path

from superbench.common.utils import SuperBenchLogger, logger


class SuperBenchExecutor():
    """SuperBench executor class."""
    def __init__(self, sb_config, docker_config, output_dir):
        """Initilize.

        Args:
            sb_config (DictConfig): SuperBench config object.
            docker_config (DictConfig): Docker config object.
            output_dir (str): Dir for output.
        """
        self._sb_config = sb_config
        self._docker_config = docker_config
        self._output_dir = output_dir
        SuperBenchLogger.add_handler(logger.logger, filename=str(Path(self._output_dir) / 'sb-exec.log'))

    def exec(self):
        """Run the SuperBench benchmarks locally.

        Raises:
            NotImplementedError: Not implemented yet.
        """
        logger.info(self._sb_config)
        raise NotImplementedError
