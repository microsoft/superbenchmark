# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner."""

import json
import random
from pathlib import Path
from collections import defaultdict

from natsort import natsorted
from joblib import Parallel, delayed
from omegaconf import ListConfig, OmegaConf

from superbench.common.utils import SuperBenchLogger, logger
from superbench.runner.ansible import AnsibleClient
from superbench.benchmarks import ReduceType, Reducer


class SuperBenchRunner():
    """SuperBench runner class."""
    def __init__(self, sb_config, docker_config, ansible_config, sb_output_dir):
        """Initilize.

        Args:
            sb_config (DictConfig): SuperBench config object.
            docker_config (DictConfig): Docker config object.
            ansible_config (DictConfig): Ansible config object.
            sb_output_dir (str): SuperBench output directory.
        """
        self._sb_config = sb_config
        self._docker_config = docker_config
        self._ansible_config = ansible_config
        self._sb_output_dir = sb_output_dir
        self._output_path = Path(sb_output_dir).expanduser().resolve()
        self._ansible_client = AnsibleClient(ansible_config)

        self.__set_logger('sb-run.log')
        logger.info('Runner uses config: %s.', self._sb_config)
        logger.info('Runner writes to: %s.', str(self._output_path))

        self._sb_benchmarks = self._sb_config.superbench.benchmarks
        self.__validate_sb_config()
        self._sb_enabled_benchmarks = self.__get_enabled_benchmarks()
        logger.info('Runner will run: %s', self._sb_enabled_benchmarks)

    def __set_logger(self, filename):
        """Set logger and add file handler.

        Args:
            filename (str): Log file name.
        """
        SuperBenchLogger.add_handler(logger.logger, filename=str(self._output_path / filename))

    def __validate_sb_config(self):    # noqa: C901
        """Validate SuperBench config object.

        Raise:
            InvalidConfigError: If input config is invalid.
        """
        # TODO: add validation and defaulting
        if not self._sb_config.superbench.env:
            self._sb_config.superbench.env = {}
        for name in self._sb_benchmarks:
            if not self._sb_benchmarks[name].modes:
                self._sb_benchmarks[name].modes = []
            for idx, mode in enumerate(self._sb_benchmarks[name].modes):
                if mode.name == 'local':
                    if not mode.proc_num:
                        self._sb_benchmarks[name].modes[idx].proc_num = 1
                    if not mode.prefix:
                        self._sb_benchmarks[name].modes[idx].prefix = ''
                elif mode.name == 'torch.distributed':
                    if not mode.proc_num:
                        self._sb_benchmarks[name].modes[idx].proc_num = 8
                elif mode.name == 'mpi':
                    if not mode.mca:
                        self._sb_benchmarks[name].modes[idx].mca = {
                            'pml': 'ob1',
                            'btl': '^openib',
                            'btl_tcp_if_exclude': 'lo,docker0',
                            'coll_hcoll_enable': 0,
                        }
                    if not mode.env:
                        self._sb_benchmarks[name].modes[idx].env = {}
                    for key in ['PATH', 'LD_LIBRARY_PATH', 'SB_MICRO_PATH']:
                        self._sb_benchmarks[name].modes[idx].env.setdefault(key, None)

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

    def __get_mode_command(self, benchmark_name, mode):
        """Get runner command for given mode.

        Args:
            benchmark_name (str): Benchmark name.
            mode (DictConfig): Runner mode.

        Return:
            str: Runner command.
        """
        exec_command = ('sb exec --output-dir {output_dir} -c sb.config.yaml -C superbench.enable={name}').format(
            name=benchmark_name,
            output_dir=self._sb_output_dir,
        )
        mode_command = exec_command
        if mode.name == 'local':
            mode_command = '{prefix} {command}'.format(
                prefix=mode.prefix.format(proc_rank=mode.proc_rank, proc_num=mode.proc_num),
                command=exec_command,
            )
            mode_command = f'PROC_RANK={mode.proc_rank} {mode_command.strip()}'
        elif mode.name == 'torch.distributed':
            # TODO: replace with torch.distributed.run in v1.9
            # TODO: only supports node_num=1 and node_num=all currently
            torch_dist_params = '' if mode.node_num == 1 else \
                '--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT '
            mode_command = (
                f'python3 -m torch.distributed.launch'
                f' --use_env --no_python --nproc_per_node={mode.proc_num} {torch_dist_params}{exec_command}'
                f' superbench.benchmarks.{benchmark_name}.parameters.distributed_impl=ddp'
                f' superbench.benchmarks.{benchmark_name}.parameters.distributed_backend=nccl'
            )
        elif mode.name == 'mpi':
            mode_command = (
                'mpirun '    # use default OpenMPI in image
                '-tag-output '    # tag mpi output with [jobid,rank]<stdout/stderr> prefix
                '-allow-run-as-root '    # allow mpirun to run when executed by root user
                '-hostfile hostfile '    # use prepared hostfile
                '-map-by ppr:{proc_num}:node '    # launch {proc_num} processes on each node
                '-bind-to numa '    # bind processes to numa
                '{mca_list} {env_list} {command}'
            ).format(
                proc_num=mode.proc_num,
                mca_list=' '.join(f'-mca {k} {v}' for k, v in mode.mca.items()),
                env_list=' '.join(f'-x {k}={v}' if v else f'-x {k}' for k, v in mode.env.items()),
                command=exec_command,
            )
        else:
            logger.warning('Unknown mode %s.', mode.name)
        return mode_command.strip()

    def deploy(self):    # pragma: no cover
        """Deploy SuperBench environment."""
        logger.info('Preparing SuperBench environment.')
        extravars = {
            'ssh_port': random.randint(1 << 14, (1 << 15) - 1),
            'output_dir': str(self._output_path),
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
        self._ansible_client.run(self._ansible_client.get_playbook_config('deploy.yaml', extravars=extravars))

    def check_env(self):    # pragma: no cover
        """Check SuperBench environment."""
        logger.info('Checking SuperBench environment.')
        OmegaConf.save(config=self._sb_config, f=str(self._output_path / 'sb.config.yaml'))
        self._ansible_client.run(
            self._ansible_client.get_playbook_config(
                'check_env.yaml',
                extravars={
                    'output_dir': str(self._output_path),
                    'env': '\n'.join(f'{k}={v}' for k, v in self._sb_config.superbench.env.items()),
                }
            )
        )

    def fetch_results(self):    # pragma: no cover
        """Fetch benchmark results on all nodes."""
        try:
            (self._output_path / 'nodes').mkdir(mode=0o755, parents=True, exist_ok=True)
        except Exception:
            logger.exception('Failed to create directory %s.', str(self._output_path / 'nodes'))
            raise
        self._ansible_client.run(
            self._ansible_client.get_playbook_config(
                'fetch_results.yaml',
                extravars={
                    'sb_output_dir': self._sb_output_dir,
                    'absolute_output_dir': str(self._output_path),
                }
            )
        )

    def __create_results_summary(self):    # pragma: no cover
        """Create the result summary file of all nodes."""
        all_results = list()
        for node_path in (self._output_path / 'nodes').glob('*'):
            if not node_path.is_dir():
                continue
            results_summary = self.__create_single_node_summary(node_path)
            results_summary['node'] = node_path.name
            all_results.append(results_summary)

        with (self._output_path / 'results-summary.jsonl').open(mode='w') as f:
            for result in all_results:
                json.dump(result, f)
                f.write('\n')

    def __create_single_node_summary(self, node_path):    # pragma: no cover
        """Create the result summary file of single node.

        Args:
            node_path (Path): The Path instance of node directory.

        Returns:
            dict: Result summary of single node.
        """
        results_summary = dict()
        reduce_ops = dict()
        file_list = [Path(f) for f in natsorted([str(f) for f in node_path.glob('**/results.json')])]
        for results_file in file_list:
            with results_file.open() as f:
                try:
                    results = json.load(f)
                except ValueError:
                    logger.error('Invalid JSON file: {}'.format(results_file))
                    continue

                for result in results:
                    benchmark_name = result['name']
                    if results_file.parts[-3].endswith('_models'):
                        benchmark_name = '{}/{}'.format(results_file.parts[-3], result['name'])
                    if benchmark_name not in results_summary:
                        results_summary[benchmark_name] = defaultdict(list)
                    for metric in result['result']:
                        metric_name = '{}/{}'.format(benchmark_name, metric)
                        if metric_name not in reduce_ops:
                            reduce_ops[metric_name] = result['reduce_op'][metric]
                        elif reduce_ops[metric_name] != result['reduce_op'][metric]:
                            logger.error('Inconsistent reduce type for metric: {}'.format(metric_name))
                            continue

                        results_summary[benchmark_name][metric].append(result['result'][metric])

        results_summary = self.__merge_all_metrics(results_summary, reduce_ops)
        with (node_path / 'results-summary.json').open(mode='w') as f:
            json.dump(results_summary, f, indent=2)

        return results_summary

    def __merge_all_metrics(self, results_summary, reduce_ops):
        """Merge metrics of all benchmarks in one node.

        Args:
            results_summary (dict): Summarized result of one node.
            reduce_ops (dict): The reduce type of each metric.

        Returns:
            dict: Flattened result with metric as key.
        """
        metrics_summary = dict()
        for benchmark_name in results_summary:
            for metric in results_summary[benchmark_name]:
                metric_name = '{}/{}'.format(benchmark_name, metric)
                if metric_name not in reduce_ops or (
                    reduce_ops[metric_name] is not None and reduce_ops[metric_name] not in ReduceType.get_values()
                ):
                    logger.error('Unknown reduce type for metric: {}'.format(metric_name))
                    continue

                if reduce_ops[metric_name] is not None:
                    reduce_func = Reducer.get_reduce_func(ReduceType(reduce_ops[metric_name]))
                    values = [reduce_func(list(result)) for result in zip(*results_summary[benchmark_name][metric])]
                    for run_count in range(len(values)):
                        if len(values) > 1:
                            metric_name = '{}/{}/{}'.format(benchmark_name, run_count, metric)
                        else:
                            metric_name = '{}/{}'.format(benchmark_name, metric)
                        metrics_summary[metric_name] = values[run_count]
                else:
                    for rank in range(len(results_summary[benchmark_name][metric])):
                        for run_count in range(len(results_summary[benchmark_name][metric][rank])):
                            if len(results_summary[benchmark_name][metric][rank]) > 1:
                                metric_name = '{}/{}/{}:{}'.format(benchmark_name, run_count, metric, rank)
                            else:
                                metric_name = '{}/{}:{}'.format(benchmark_name, metric, rank)
                            metrics_summary[metric_name] = results_summary[benchmark_name][metric][rank][run_count]

        return metrics_summary

    def _run_proc(self, benchmark_name, mode, vars):
        """Run the process.

        Args:
            benchmark_name (str): Benchmark name.
            mode (DictConfig): Runner mode.
            vars (dict): Process variables.

        Returns:
            int: Process return code.
        """
        mode.update(vars)
        logger.info('Runner is going to run %s in %s mode, proc rank %d.', benchmark_name, mode.name, mode.proc_rank)
        ansible_runner_config = self._ansible_client.get_shell_config(
            (
                'docker exec sb-workspace bash -c '
                "'set -o allexport && source sb.env && set +o allexport && {command}'"
            ).format(command=self.__get_mode_command(benchmark_name, mode))
        )
        if mode.name == 'mpi':
            ansible_runner_config = self._ansible_client.update_mpi_config(ansible_runner_config)
        rc = self._ansible_client.run(ansible_runner_config, sudo=True)
        return rc

    def run(self):
        """Run the SuperBench benchmarks distributedly."""
        self.check_env()
        for benchmark_name in self._sb_benchmarks:
            if benchmark_name not in self._sb_enabled_benchmarks:
                continue
            benchmark_config = self._sb_benchmarks[benchmark_name]
            for mode in benchmark_config.modes:
                if mode.name == 'local':
                    Parallel(n_jobs=mode.proc_num if mode.parallel else 1)(
                        delayed(self._run_proc)(benchmark_name, mode, {
                            'proc_rank': proc_rank
                        }) for proc_rank in range(mode.proc_num)
                    )
                elif mode.name == 'torch.distributed' or mode.name == 'mpi':
                    self._run_proc(benchmark_name, mode, {'proc_rank': 0})
                else:
                    logger.warning('Unknown mode %s.', mode.name)
            self.fetch_results()

        self.__create_results_summary()
