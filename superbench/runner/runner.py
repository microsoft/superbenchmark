# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner."""

import random
import multiprocessing
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
        self._sb_enabled_benchmarks = self.__get_enabled_benchmarks()
        logger.info('Runner will run: %s', self._sb_enabled_benchmarks)

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
            mode (DictConfig): Runner mode.
            exec_command (str): Executor command.

        Return:
            str: Runner command.
        """
        mode_command = exec_command
        if mode.name == 'local':
            mode_command = '{prefix} {command}'.format(
                prefix=(mode.prefix or '').format(proc_rank=mode.proc_rank, proc_num=mode.proc_num or 1),
                command=exec_command
            )
        elif mode.name == 'torch.distributed':
            # TODO: replace with torch.distributed.run in v1.9
            # TODO: only supports node_num=1 and node_num=all currently
            mode_command = (
                'python3 -m torch.distributed.launch '
                '--use_env --no_python --nproc_per_node={proc_num} '
                '--nnodes={node_num} --node_rank=$NODE_RANK '
                '--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
                '{command}'
            ).format(
                proc_num=mode.proc_num or 8, node_num=1 if mode.node_num == 1 else '$NNODES', command=exec_command
            )
        return mode_command.strip()

    def deploy(self):    # pragma: no cover
        """Deploy SuperBench environment."""
        logger.info('Preparing SuperBench environment.')
        extravars = {
            'ssh_port': random.randint(1 << 14, (1 << 15) - 1),
            'output_dir': self._output_dir,
            'docker_image': self._docker_config.image,
            'gpu_vendor': 'nvidia',
        }
        if bool(self._docker_config.username) and bool(self._docker_config.password):
            extravars.update(
                {
                    'docker_registry': self._docker_config.registry,
                    'docker_username': self._docker_config.username,
                    'docker_password': self._docker_config.password,
                }
            )
        self._ansible_client.run(self._ansible_client.get_playbook_config('deploy.yaml', extravars=extravars))

    def check_env(self):    # pragma: no cover
        """Check SuperBench environment."""
        logger.info('Checking SuperBench environment.')
        OmegaConf.save(config=self._sb_config, f=str(Path(self._output_dir) / 'sb.config.yaml'))
        self._ansible_client.run(
            self._ansible_client.get_playbook_config('check_env.yaml', extravars={'output_dir': self._output_dir})
        )

    def run(self):
        """Run the SuperBench benchmarks distributedly."""
        self.check_env()
        runner_command = (
            'docker exec sb-workspace bash -c '
            '"set -o allexport && source sb.env && set +o allexport && {}"'
        )
        for benchmark_name in self._sb_benchmarks:
            if benchmark_name not in self._sb_enabled_benchmarks:
                continue
            benchmark_config = self._sb_benchmarks[benchmark_name]
            for mode in benchmark_config.modes or []:
                if mode.name == 'local':
                    logger.info('Runner is going to run %s.', benchmark_name)

                    def run_proc(vars, rets):
                        mode.update(vars)
                        rc = self._ansible_client.run(
                            self._ansible_client.get_shell_config(
                                runner_command.format(
                                    self.__get_mode_command(
                                        mode, ('sb exec -c sb.config.yaml -C '
                                               'superbench.enable={name}').format(name=benchmark_name)
                                    )
                                )
                            ),
                            sudo=True
                        )
                        rets[vars['proc_rank']] = rc

                    jobs = []
                    rets = multiprocessing.Manager().dict()
                    for proc_rank in range(mode.proc_num or 1):
                        proc = multiprocessing.Process(target=run_proc, args=({'proc_rank': proc_rank}, rets))
                        jobs.append(proc)
                    for proc in jobs:
                        proc.start()
                    for proc in jobs:
                        proc.join()

                elif mode.name == 'torch.distributed':
                    logger.info('Runner is going to run %s.', benchmark_name)
                    self._ansible_client.run(
                        self._ansible_client.get_shell_config(
                            runner_command.format(
                                self.__get_mode_command(
                                    mode, (
                                        'sb exec -c sb.config.yaml -C '
                                        'superbench.enable={name} '
                                        'superbench.benchmarks.{name}.parameters.distributed_impl=ddp '
                                        'superbench.benchmarks.{name}.parameters.distributed_backend=nccl'
                                    ).format(name=benchmark_name)
                                )
                            )
                        ),
                        sudo=True
                    )
