# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Executor."""

import os
import json
from pathlib import Path

from omegaconf import ListConfig

from superbench.benchmarks import Platform, Framework, BenchmarkRegistry
from superbench.common.utils import SuperBenchLogger, logger, rotate_dir, stdout_logger
from superbench.common.devices import GPU
from superbench.monitor import Monitor


class SuperBenchExecutor():
    """SuperBench executor class."""
    def __init__(self, sb_config, sb_output_dir):
        """Initilize.

        Args:
            sb_config (DictConfig): SuperBench config object.
            sb_output_dir (str): SuperBench output directory.
        """
        self._sb_config = sb_config
        self._sb_output_dir = sb_output_dir
        self._output_path = Path(sb_output_dir).expanduser().resolve()

        self.__set_logger('sb-exec.log')
        self.__set_stdout_logger(self._output_path / 'sb-bench.log')
        logger.debug('Executor uses config: %s.', self._sb_config)
        logger.debug('Executor writes to: %s.', str(self._output_path))

        self.__validate_sb_config()
        self._sb_monitor_config = self._sb_config.superbench.monitor
        self._sb_benchmarks = self._sb_config.superbench.benchmarks
        self._sb_enabled = self.__get_enabled_benchmarks()
        logger.debug('Executor will execute: %s', self._sb_enabled)

    def __set_logger(self, filename):
        """Set logger and add file handler.

        Args:
            filename (str): Log file name.
        """
        SuperBenchLogger.add_handler(logger.logger, filename=str(self._output_path / filename))

    def __set_stdout_logger(self, filename):
        """Set stdout logger and redirect logs and stdout into the file.

        Args:
            filename (str): Log file name.
        """
        stdout_logger.add_file_handler(filename)
        stdout_logger.start(self.__get_rank_id())
        SuperBenchLogger.add_handler(logger.logger, filename=filename)

    def __validate_sb_config(self):
        """Validate SuperBench config object.

        Raise:
            InvalidConfigError: If input config is invalid.
        """
        # TODO: add validation

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
        # TODO: may exist order issue
        return [k for k, v in self._sb_benchmarks.items() if v.enable]

    def __get_platform(self):
        """Detect runninng platform by environment."""
        try:
            gpu = GPU()
            if gpu.vendor == 'nvidia':
                return Platform.CUDA
            elif gpu.vendor == 'amd':
                return Platform.ROCM
            elif gpu.vendor == 'amd-graphics' or gpu.vendor == 'nvidia-graphics':
                return Platform.DIRECTX
        except Exception as e:
            logger.error(e)
        return Platform.CPU

    def __get_arguments(self, parameters):
        """Get command line arguments for argparse.

        Args:
            parameters (DictConfig): Parameters config dict.

        Return:
            str: Command line arguments.
        """
        argv = []
        if not parameters:
            return ''
        for name, val in parameters.items():
            if val is None:
                continue
            if isinstance(val, bool):
                if val:
                    argv.append('--{}'.format(name))
            elif isinstance(val, (str, int, float)):
                argv.append('--{} {}'.format(name, val))
            elif isinstance(val, (list, ListConfig)):
                argv.append('--{} {}'.format(name, ' '.join(val)))
        return ' '.join(argv)

    def __exec_benchmark(self, benchmark_full_name, context):
        """Launch benchmark for context.

        Args:
            benchmark_full_name (str): Benchmark full name.
            context (BenchmarkContext): Benchmark context to launch.

        Return:
            dict: Benchmark result.
        """
        try:
            benchmark = BenchmarkRegistry.launch_benchmark(context)
            if benchmark:
                logger.info(
                    'benchmark: %s, return code: %s, result: %s.', benchmark.name, benchmark.return_code,
                    benchmark.result
                )
                if benchmark.return_code.value == 0:
                    logger.info('Executor succeeded in %s.', benchmark_full_name)
                else:
                    logger.error('Executor failed in %s.', benchmark_full_name)
                result = json.loads(benchmark.serialized_result)
                result['name'] = benchmark_full_name
                return result
            else:
                logger.error('Executor failed in %s, invalid context.', benchmark_full_name)
        except Exception as e:
            logger.error(e)
            logger.error('Executor failed in %s.', benchmark_full_name)
        return None

    def __get_rank_id(self):
        """Get rank ID for current process.

        Return:
            int: Rank ID.
        """
        for rank_env in ['PROC_RANK', 'LOCAL_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK']:
            if os.getenv(rank_env):
                return int(os.getenv(rank_env))

        return 0

    def __get_benchmark_dir(self, benchmark_name):
        """Get output directory for benchmark's current rank.

        Args:
            benchmark_name (str): Benchmark name.

        Return:
            Path: output directory.
        """
        return self._output_path / 'benchmarks' / benchmark_name / ('rank' + str(self.__get_rank_id()))

    def __create_benchmark_dir(self, benchmark_name):
        """Create output directory for benchmark.

        Args:
            benchmark_name (str): Benchmark name.
        """
        rotate_dir(self.__get_benchmark_dir(benchmark_name))
        try:
            self.__get_benchmark_dir(benchmark_name).mkdir(mode=0o755, parents=True, exist_ok=True)
        except Exception:
            logger.exception('Failed to create output directory for benchmark %s.', benchmark_name)
            raise

    def __write_benchmark_results(self, benchmark_name, benchmark_results):
        """Write benchmark results.

        Args:
            benchmark_name (str): Benchmark name.
            benchmark_results (dict): Benchmark results.
        """
        with (self.__get_benchmark_dir(benchmark_name) / 'results.json').open(mode='w') as f:
            json.dump(benchmark_results, f, indent=2)

    def __get_monitor_path(self, benchmark_name):
        """Get the output file path for the monitor.

        Args:
            benchmark_name (str): Benchmark name.

        Return:
            str: monitor output file path.
        """
        return f'{self.__get_benchmark_dir(benchmark_name) / "monitor.jsonl"}'

    def exec(self):
        """Run the SuperBench benchmarks locally."""
        for benchmark_name in self._sb_benchmarks:
            if benchmark_name not in self._sb_enabled:
                continue
            benchmark_config = self._sb_benchmarks[benchmark_name]
            benchmark_results = list()
            self.__create_benchmark_dir(benchmark_name)
            cwd = os.getcwd()
            os.chdir(self.__get_benchmark_dir(benchmark_name))

            monitor = None
            if self.__get_rank_id() == 0 and self._sb_monitor_config and self._sb_monitor_config.enable:
                if self.__get_platform() == Platform.CUDA:
                    monitor = Monitor(
                        None, int(self._sb_monitor_config.sample_duration or 10),
                        int(self._sb_monitor_config.sample_interval or 1), self.__get_monitor_path(benchmark_name)
                    )
                    monitor.start()
                else:
                    logger.warning('Monitor can not support ROCM/CPU platform.')

            benchmark_real_name = benchmark_name.split(':')[0]
            for framework in benchmark_config.frameworks or [Framework.NONE.value]:
                if benchmark_real_name == 'model-benchmarks' or (
                    ':' not in benchmark_name and benchmark_name.endswith('_models')
                ):
                    for model in benchmark_config.models:
                        full_name = f'{benchmark_name}/{framework}-{model}'
                        logger.info('Executor is going to execute %s.', full_name)
                        context = BenchmarkRegistry.create_benchmark_context(
                            model,
                            platform=self.__get_platform(),
                            framework=Framework(framework.lower()),
                            parameters=self.__get_arguments(benchmark_config.parameters)
                        )
                        result = self.__exec_benchmark(full_name, context)
                        benchmark_results.append(result)
                else:
                    full_name = benchmark_name
                    logger.info('Executor is going to execute %s.', full_name)
                    context = BenchmarkRegistry.create_benchmark_context(
                        benchmark_real_name,
                        platform=self.__get_platform(),
                        framework=Framework(framework.lower()),
                        parameters=self.__get_arguments(benchmark_config.parameters)
                    )
                    result = self.__exec_benchmark(full_name, context)
                    benchmark_results.append(result)

            if monitor:
                monitor.stop()
            stdout_logger.stop()
            self.__write_benchmark_results(benchmark_name, benchmark_results)
            os.chdir(cwd)
