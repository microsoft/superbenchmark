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

    def __get_mode_command(self, mode, exec_command):
        """Get runner command for given mode.

        Args:
            mode (str): Runner mode.
            exec_command (str): Executor command.

        Return:
            str: Runner command.
        """
        if mode == 'torch.distributed':
            # TODO: replace with torch.distributed.run in v1.9
            return (
                'python3 -m torch.distributed.launch '
                '--use_env --no_python --nproc_per_node=8 '
                '--nnodes=$NNODES --node_rank=$NODE_RANK '
                '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                '{}'
            ).format(exec_command)
        return exec_command

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
        self.check_env()
        runner_command = (
            'docker exec sb-workspace bash -c '
            '"set -o allexport && source sb.env && set +o allexport && {}"'
        )
        for benchmark_name in self._sb_benchmarks:
            if benchmark_name not in self._sb_enabled:
                continue
            benchmark_config = self._sb_benchmarks[benchmark_name]
            if benchmark_config.mode == 'torch.distributed':
                logger.info('Runner is going to run %s.', benchmark_name)
                self._ansible_client.run_shell(
                    runner_command.format(
                        self.__get_mode_command(
                            benchmark_config.mode, (
                                'sb exec -c sb.config.yaml -C '
                                'superbench.enable={name} '
                                'superbench.benchmarks.{name}.parameters.distributed_impl=ddp '
                                'superbench.benchmarks.{name}.parameters.distributed_backend=nccl'
                            ).format(name=benchmark_name)
                        )
                    ),
                    sudo=True
                )
