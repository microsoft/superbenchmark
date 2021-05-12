# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner."""

import random
from pathlib import Path

from omegaconf import ListConfig, OmegaConf

from superbench.common.utils import SuperBenchLogger, logger
from superbench.runner.ansible import AnsibleClient


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
        self._ansible_client = AnsibleClient(ansible_config)

        self.__set_logger('sb-run.log')
        logger.info('Runner uses config: %s.', self._sb_config)
        logger.info('Runner writes to: %s.', self._output_dir)

        self._sb_benchmarks = self._sb_config.superbench.benchmarks
        self._sb_enabled = self.__get_enabled_benchmarks()
        logger.info('Runner will run: %s', self._sb_enabled)

    def __set_logger(self, filename):
        """Set logger and add file handler.

        Args:
            filename (str): Log file name.
        """
        SuperBenchLogger.add_handler(logger.logger, filename=str(Path(self._output_dir) / filename))

    def __get_enabled_benchmarks(self):
        """Get enabled benchmarks list.

        Return:
            list: List of benchmarks which will be executed.
        """
        if self._sb_config.superbench.enable:
            if isinstance(self._sb_config.superbench.enable, str):
                return [self._sb_config.superbench.enable]
            elif isinstance(self._sb_config.superbench.enable, (list, ListConfig)):
                return list(self._sb_config.superbench.enable)
        return [k for k, v in self._sb_benchmarks.items() if v.enable]

    def deploy(self):
        """Deploy SuperBench environment."""
        logger.info('Preparing SuperBench environment.')
        extravars = {
            'ssh_port': random.randint(1 << 14, (1 << 15) - 1),
            'output_dir': self._output_dir,
            'docker_image': self._docker_config.image,
        }
        if bool(self._docker_config.username) and bool(self._docker_config.password):
            extravars.update(
                {
                    'docker_registry': self._docker_config.registry,
                    'docker_username': self._docker_config.username,
                    'docker_password': self._docker_config.password,
                }
            )
        self._ansible_client.run_playbook('deploy.yaml', extravars=extravars)

    def check_env(self):
        """Check SuperBench environment."""
        logger.info('Checking SuperBench environment.')
        OmegaConf.save(config=self._sb_config, f=str(Path(self._output_dir) / 'sb.config.yaml'))
        self._ansible_client.run_playbook('check_env.yaml', extravars={'output_dir': self._output_dir})

    def run(self):
        """Run the SuperBench benchmarks distributedly.

        Raises:
            NotImplementedError: Not implemented yet.
        """
        logger.info(self._sb_config)
        logger.error('Work in progress, not implemented yet.')
        pass
