# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench Executor."""

import json
from pathlib import Path

from omegaconf import ListConfig

from superbench.benchmarks import Platform, Framework, BenchmarkRegistry
from superbench.common.utils import SuperBenchLogger, logger


class SuperBenchExecutor():
    """SuperBench executor class."""
    def __init__(self, sb_config, output_dir):
        """Initilize.

        Args:
            sb_config (DictConfig): SuperBench config object.
            output_dir (str): Dir for output.
        """
        self._sb_config = sb_config
        self._output_dir = output_dir

        self.__set_logger('sb-exec.log')
        logger.info('Executor uses config: %s.', self._sb_config)
        logger.info('Executor writes to: %s.', self._output_dir)

        self.__validate_sb_config()
        self._sb_benchmarks = self._sb_config.superbench.benchmarks
        self._sb_enabled = self.__get_enabled_benchmarks()
        logger.info('Executor will execute: %s', self._sb_enabled)

    def __set_logger(self, filename):
        """Set logger and add file handler.

        Args:
            filename (str): Log file name.
        """
        SuperBenchLogger.add_handler(logger.logger, filename=str(Path(self._output_dir) / filename))

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
        # TODO: check devices and env vars
        return Platform.CUDA

    def __get_arguments(self, parameters):
        """Get command line arguments for argparse.

        Args:
            parameters (DictConfig): Parameters config dict.

        Return:
            str: Command line arguments.
        """
        argv = []
        for name, val in parameters.items():
            if val is None:
                continue
            if isinstance(val, (str, int, float)):
                argv.append('--{} {}'.format(name, val))
            elif isinstance(val, (list, ListConfig)):
                argv.append('--{} {}'.format(name, ' '.join(val)))
            elif isinstance(val, bool) and val:
                argv.append('--{}'.format(name))
        return ' '.join(argv)

    def __exec_benchmark(self, context, log_suffix):
        """Launch benchmark for context.

        Args:
            context (BenchmarkContext): Benchmark context to launch.
            log_suffix (str): Log string suffix.

        Return:
            dict: Benchmark results.
        """
        try:
            benchmark = BenchmarkRegistry.launch_benchmark(context)
            if benchmark:
                logger.info(
                    'benchmark: %s, return code: %s, result: %s.', benchmark.name, benchmark.return_code,
                    benchmark.result
                )
                if benchmark.return_code.value == 0:
                    logger.info('Executor succeeded in %s.', log_suffix)
                else:
                    logger.error('Executor failed in %s.', log_suffix)
                return benchmark.result
            else:
                logger.error('Executor failed in %s, invalid context.', log_suffix)
        except Exception as e:
            logger.error(e)
            logger.error('Executor failed in %s.', log_suffix)
        return None

    def __get_benchmark_dir(self, benchmark_name, create=False):
        """Get output directory for benchmark.

        Args:
            benchmark_name (str): Benchmark name.
            create (bool): Create the directory or not.

        Return:
            str: Benchmark output directory.
        """
        benchmark_output_dir = Path(self._output_dir, 'benchmarks', benchmark_name)
        if create:
            benchmark_output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
        return str(benchmark_output_dir)

    def __write_benchmark_results(self, benchmark_name, benchmark_results):
        """Write benchmark results.

        Args:
            benchmark_name (str): Benchmark name.
            benchmark_results (dict): Benchmark results.
        """
        with Path(self.__get_benchmark_dir(benchmark_name), 'results.json').open(mode='w') as f:
            json.dump(benchmark_results, f, indent=2)

    def exec(self):
        """Run the SuperBench benchmarks locally."""
        for benchmark_name in self._sb_benchmarks:
            if benchmark_name not in self._sb_enabled:
                continue
            benchmark_config = self._sb_benchmarks[benchmark_name]
            benchmark_results = {}
            self.__get_benchmark_dir(benchmark_name, create=True)
            for framework in benchmark_config.frameworks or [Framework.NONE]:
                if benchmark_name.endswith('_models'):
                    for model in benchmark_config.models:
                        log_suffix = 'model-benchmark {}: {}/{}'.format(benchmark_name, framework, model)
                        logger.info('Executor is going to execute %s.', log_suffix)
                        context = BenchmarkRegistry.create_benchmark_context(
                            model,
                            platform=self.__get_platform(),
                            framework=Framework(framework.lower()),
                            parameters=self.__get_arguments(benchmark_config.parameters)
                        )
                        result = self.__exec_benchmark(context, log_suffix)
                        if framework != Framework.NONE:
                            benchmark_results['{}/{}'.format(framework, model)] = result
                        else:
                            benchmark_results[model] = result
                else:
                    log_suffix = 'micro-benchmark {}'.format(benchmark_name)
                    logger.info('Executor is going to execute %s.', log_suffix)
                    context = BenchmarkRegistry.create_benchmark_context(
                        benchmark_name,
                        platform=self.__get_platform(),
                        framework=Framework(framework.lower()),
                        parameters=self.__get_arguments(benchmark_config.parameters)
                    )
                    result = self.__exec_benchmark(context, log_suffix)
                    if framework != Framework.NONE:
                        benchmark_results[framework] = result
                    else:
                        benchmark_results = result
            self.__write_benchmark_results(benchmark_name, benchmark_results)
