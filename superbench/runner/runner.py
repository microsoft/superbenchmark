# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner."""

from pathlib import Path

from superbench.common.utils import SuperBenchLogger, logger


class SuperBenchRunner():
    """SuperBench runner class."""
    def __init__(self, sb_config, docker_config, ansible_config, output_dir):
        """Initilize.

        Args:
            sb_config (DictConfig): SuperBench config object.
            docker_config (DictConfig): Docker config object.
            ansible_config (DictConfig): Ansible config object.
            output_dir (str): Dir for output.
        """
        self._sb_config = sb_config
        self._docker_config = docker_config
        self._ansible_config = ansible_config
        self._output_dir = output_dir

        self.__set_logger('sb-run.log')
        logger.info('Runner uses config: %s.', self._sb_config)
        logger.info('Runner writes to: %s.', self._output_dir)

    def __set_logger(self, filename):
        """Set logger and add file handler.

        Args:
            filename (str): Log file name.
        """
        SuperBenchLogger.add_handler(logger.logger, filename=str(Path(self._output_dir) / filename))

    def run(self):
        """Run the SuperBench benchmarks distributedly.

        Raises:
            NotImplementedError: Not implemented yet.
        """
        logger.info(self._sb_config)
        logger.error('Work in progress, not implemented yet.')
        pass
