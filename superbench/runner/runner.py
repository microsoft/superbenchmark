# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Runner."""

import os
import sys
import json
import random
import signal
from pathlib import Path
from pprint import pformat
from collections import defaultdict

import jsonlines
from natsort import natsorted
from joblib import Parallel, delayed
from omegaconf import ListConfig, OmegaConf
import nvtx

from superbench.common.utils import SuperBenchLogger, logger, gen_ibstat, gen_traffic_pattern_host_groups
from superbench.common.utils.lazy_import import LazyImport
from superbench.benchmarks import ReduceType, Reducer
from superbench.monitor import MonitorRecord

AnsibleClient = LazyImport('superbench.runner.ansible', 'AnsibleClient')


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
        logger.info('Runner uses config: %s.', pformat(OmegaConf.to_container(self._sb_config, resolve=True)))
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
                if not mode.env:
                    self._sb_benchmarks[name].modes[idx].env = {}
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
                    for key in ['PATH', 'LD_LIBRARY_PATH', 'SB_MICRO_PATH', 'SB_WORKSPACE']:
                        self._sb_benchmarks[name].modes[idx].env.setdefault(key, None)
                    if mode.pattern:
                        if mode.pattern.type == 'topo-aware' and not mode.pattern.ibstat:
                            self._sb_benchmarks[name].modes[idx].pattern.ibstat = gen_ibstat(
                                self._ansible_config, str(self._output_path / 'ibstate_file.txt')
                            )

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

    def __get_mode_command(self, benchmark_name, mode, timeout=None):
        """Get runner command for given mode.

        Args:
            benchmark_name (str): Benchmark name.
            mode (DictConfig): Runner mode.
            timeout (int): The timeout value in seconds.
            host_list (list): The specified Host node list.

        Return:
            str: Runner command.
        """
        exec_command = ('sb exec --output-dir {output_dir} -c sb.config.yaml -C superbench.enable={name}').format(
            name=benchmark_name,
            output_dir=self._sb_output_dir,
        )
        if timeout is not None:
            exec_command = 'timeout {timeout} {command}'.format(timeout=timeout, command=exec_command)

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
                f'torchrun'
                f' --no_python --nproc_per_node={mode.proc_num} {torch_dist_params}{exec_command}'
                f' superbench.benchmarks.{benchmark_name}.parameters.distributed_impl=ddp'
                f' superbench.benchmarks.{benchmark_name}.parameters.distributed_backend=nccl'
            )
        elif mode.name == 'mpi':
            mode_command = (
                'mpirun '    # use default OpenMPI in image
                '-tag-output '    # tag mpi output with [jobid,rank]<stdout/stderr> prefix
                '-allow-run-as-root '    # allow mpirun to run when executed by root user
                '{host_list} '    # use prepared hostfile or specify nodes and launch {proc_num} processes on each node
                '-bind-to numa '    # bind processes to numa
                '{mca_list} {env_list} {command}'
            ).format(
                host_list=f'-host localhost:{mode.proc_num}' if mode.node_num == 1 else
                f'-hostfile hostfile -map-by ppr:{mode.proc_num}:node' if mode.host_list is None else '-host ' +
                ','.join(f'{host}:{mode.proc_num}' for host in mode.host_list),
                mca_list=' '.join(f'-mca {k} {v}' for k, v in mode.mca.items()),
                env_list=' '.join(
                    f'-x {k}={str(v).format(proc_rank=mode.proc_rank, proc_num=mode.proc_num)}'
                    if isinstance(v, str) else f'-x {k}' for k, v in mode.env.items()
                ),
                command=exec_command,
            )
        else:
            logger.warning('Unknown mode %s.', mode.name)
        return mode_command.strip()

    def get_failure_count(self):
        """Get failure count during Ansible run.

        Return:
            int: Failure count.
        """
        return self._ansible_client.failure_count

    def deploy(self):    # pragma: no cover
        """Deploy SuperBench environment."""
        logger.info('Preparing SuperBench environment.')
        extravars = {
            'ssh_port': random.randint(1 << 14, (1 << 15) - 1),
            'output_dir': str(self._output_path),
            'docker_image': self._docker_config.image,
            'docker_pull': bool(self._docker_config.pull),
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

    def run_sys_info(self):
        """Run the system info on all nodes."""
        self.check_env()

        logger.info('Runner is going to get node system info.')

        fcmd = "docker exec sb-workspace bash -c '{command}'"
        if self._docker_config.skip:
            fcmd = "bash -c 'cd $SB_WORKSPACE && {command}'"
        ansible_runner_config = self._ansible_client.get_shell_config(
            fcmd.format(command='sb node info --output-dir {output_dir}'.format(output_dir=self._sb_output_dir))
        )
        ansible_rc = self._ansible_client.run(ansible_runner_config, sudo=(not self._docker_config.skip))

        if ansible_rc != 0:
            self.cleanup()
        self.fetch_results()

    def check_env(self):    # pragma: no cover
        """Check SuperBench environment."""
        logger.info('Checking SuperBench environment.')
        OmegaConf.save(config=self._sb_config, f=str(self._output_path / 'sb.config.yaml'))
        self._ansible_client.run(
            self._ansible_client.get_playbook_config(
                'check_env.yaml',
                extravars={
                    'no_docker': bool(self._docker_config.skip),
                    'output_dir': str(self._output_path),
                    'env': '\n'.join(f'{k}={v}' for k, v in self._sb_config.superbench.env.items()),
                }
            )
        )

    def cleanup(self):    # pragma: no cover
        """Cleanup remaining processes on all nodes."""
        self._ansible_client.run(self._ansible_client.get_playbook_config('cleanup.yaml'))

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

    def __signal_handler(self, signum, frame):
        """Signal handler for runner.

        Args:
            signum (int): Signal number.
            frame (FrameType): Timeout frame.
        """
        if signum == signal.SIGINT or signum == signal.SIGTERM:
            logger.info('Killed by %s, exiting ...', signal.Signals(signum).name)
            self.cleanup()
            sys.exit(128 + signum)

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

    def __create_single_node_summary(self, node_path):    # pragma: no cover # noqa: C901
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
                    try:
                        benchmark_name = result['name']
                    except Exception:
                        logger.error('Invalid content in JSON file: {}'.format(results_file))
                        continue
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

        results_summary = self.__merge_benchmark_metrics(results_summary, reduce_ops)
        monitor_summary = self.__merge_monitor_metrics(node_path)
        results_summary = {**results_summary, **monitor_summary}
        with (node_path / 'results-summary.json').open(mode='w') as f:
            json.dump(results_summary, f, indent=2)

        return results_summary

    def __generate_metric_name(self, benchmark_name, metric, rank_count, run_count, curr_rank, curr_run):
        """Generate the summarized metrics name.

        The format of metric name is:
               {benchmark_name}/[{run_count}/]{metric_name}[:rank]
        [run_count] and [rank] parts are optional.

        Args:
            benchmark_name (str): The benchmark name.
            metric (str): The metric name.
            rank_count (int): The total count of rank.
            run_count (int): The total count of benchmarking.
            curr_rank (int): The current rank index.
            curr_run (int): The current run index.

        Returns:
            dict: Flattened result with metric as key.
        """
        metric_name = benchmark_name
        if run_count > 1:
            metric_name = '{}/{}'.format(metric_name, curr_run)
        metric_name = '{}/{}'.format(metric_name, metric)
        if rank_count > 1:
            metric_name = '{}:{}'.format(metric_name, curr_rank)

        return metric_name

    def __merge_benchmark_metrics(self, results_summary, reduce_ops):
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
                    for run in range(len(values)):
                        metric_name = self.__generate_metric_name(benchmark_name, metric, 1, len(values), 0, run)
                        metrics_summary[metric_name] = values[run]
                else:
                    rank_count = len(results_summary[benchmark_name][metric])
                    for rank, rank_value in enumerate(results_summary[benchmark_name][metric]):
                        run_count = len(rank_value)
                        for run, run_value in enumerate(rank_value):
                            metric_name = self.__generate_metric_name(
                                benchmark_name, metric, rank_count, run_count, rank, run
                            )
                            metrics_summary[metric_name] = run_value

        return metrics_summary

    def __merge_monitor_metrics(self, node_path):
        """Merge and summarize monitor metrics of one node.

        Args:
            node_path (Path): The Path instance of node directory.

        Returns:
            dict: Flattened result with metric as key.
        """
        metrics_summary = dict()
        all_samples = list()
        file_list = list(node_path.glob('**/monitor.jsonl'))
        for results_file in file_list:
            try:
                with jsonlines.open(results_file) as reader:
                    all_samples = list(reader)
            except BaseException as e:
                logger.error('Invalid Jsonline file: {}, error message: {}'.format(results_file, str(e)))
                continue
        all_samples = sorted(all_samples, key=lambda k: k.get('time', '0'))
        metrics_dict = dict()
        for sample in all_samples:
            for metric, value in sample.items():
                if metric not in metrics_dict:
                    metrics_dict[metric] = list()
                metrics_dict[metric].append(value)

        for metric, values in metrics_dict.items():
            prefix = metric.split(':')[0]
            for pattern, reduce_type in MonitorRecord.reduce_ops.items():
                if pattern == prefix:
                    reduce_func = Reducer.get_reduce_func(reduce_type)
                    metric_name = 'monitor/{}'.format(metric)
                    metrics_summary[metric_name] = reduce_func(values)
                    continue

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
        rng = nvtx.start_range(message="BENCHMARK", color="green")
        mode.update(vars)
        if mode.name == 'mpi' and mode.pattern:
            mode.env.update({'SB_MODE_SERIAL_INDEX': mode.serial_index, 'SB_MODE_PARALLEL_INDEX': mode.parallel_index})
        logger.info('Runner is going to run %s in %s mode, proc rank %d.', benchmark_name, mode.name, mode.proc_rank)

        timeout = self._sb_benchmarks[benchmark_name].timeout
        if isinstance(timeout, int):
            timeout = max(timeout, 60)

        env_list = '--env-file /tmp/sb.env'
        if self._docker_config.skip:
            env_list = 'set -o allexport && source /tmp/sb.env && set +o allexport'
        for k, v in mode.env.items():
            if isinstance(v, str):
                envvar = f'{k}={str(v).format(proc_rank=mode.proc_rank, proc_num=mode.proc_num)}'
                env_list += f' -e {envvar}' if not self._docker_config.skip else f' && export {envvar}'

        fcmd = "docker exec {env_list} sb-workspace bash -c '{command}'"
        if self._docker_config.skip:
            fcmd = "bash -c '{env_list} && cd $SB_WORKSPACE && {command}'"
        ansible_runner_config = self._ansible_client.get_shell_config(
            fcmd.format(env_list=env_list, command=self.__get_mode_command(benchmark_name, mode, timeout))
        )
        if mode.name == 'mpi' and mode.node_num != 1:
            ansible_runner_config = self._ansible_client.update_mpi_config(ansible_runner_config)

        if isinstance(timeout, int):
            # we do not expect timeout in ansible unless subprocess hangs
            ansible_runner_config['timeout'] = timeout + 60

        # overwrite ansible runner's default signal handler with main process's
        rc = self._ansible_client.run(
            ansible_runner_config, cancel_callback=lambda: None, sudo=(not self._docker_config.skip)
        )
        nvtx.end_range(rng)
        return rc

    def run(self):
        """Run the SuperBench benchmarks distributedly."""
        self.check_env()
        signal.signal(signal.SIGINT, self.__signal_handler)
        signal.signal(signal.SIGTERM, self.__signal_handler)
        for benchmark_name in self._sb_benchmarks:
            if benchmark_name not in self._sb_enabled_benchmarks:
                continue
            benchmark_config = self._sb_benchmarks[benchmark_name]
            for mode in benchmark_config.modes:
                ansible_rc = 0
                if mode.name == 'local':
                    rc_list = Parallel(n_jobs=mode.proc_num if mode.parallel else 1)(
                        delayed(self._run_proc)(benchmark_name, mode, {
                            'proc_rank': proc_rank
                        }) for proc_rank in range(mode.proc_num)
                    )
                    ansible_rc = sum(rc_list)
                elif mode.name == 'torch.distributed' or mode.name == 'mpi':
                    if not mode.pattern:
                        ansible_rc = self._run_proc(benchmark_name, mode, {'proc_rank': 0})
                    else:
                        if not os.path.exists(self._output_path / 'hostfile'):
                            logger.warning('No hostfile under %s.', self._output_path)
                            continue
                        with open(self._output_path / 'hostfile', 'r') as f:
                            host_list = f.read().splitlines()
                        host_groups = gen_traffic_pattern_host_groups(
                            host_list, mode.pattern, self._output_path / 'mpi_pattern.txt', benchmark_name
                        )
                        for serial_index, host_group in enumerate(host_groups):
                            para_rc_list = Parallel(n_jobs=len(host_group))(
                                delayed(self._run_proc)(
                                    benchmark_name,
                                    mode,
                                    vars={
                                        'proc_rank': 0,
                                        'host_list': host_list,
                                        'serial_index': str(serial_index),
                                        'parallel_index': str(parallel_index),
                                    }
                                ) for parallel_index, host_list in enumerate(host_group)
                            )
                            ansible_rc = ansible_rc + sum(para_rc_list)
                else:
                    logger.warning('Unknown mode %s.', mode.name)
                if ansible_rc != 0:
                    self.cleanup()
            self.fetch_results()

        self.__create_results_summary()
