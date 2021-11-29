# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Interfaces that provide access to benchmarks."""

from typing import Dict

from superbench.common.utils import logger
from superbench.benchmarks import Platform, Framework, BenchmarkContext
from superbench.benchmarks.base import Benchmark


class BenchmarkRegistry:
    """Class that minatains all benchmarks.

    Provide the following functions:
        Register new benchmark.
        Get the internal benchmark name.
        Check the validation of benchmark parameters.
        Get all configurable settings of benchmark.
        Launch one benchmark and return the result.
    """
    benchmarks: Dict[str, dict] = dict()

    @classmethod
    def register_benchmark(cls, name, class_def, parameters='', platform=None):
        """Register new benchmark, key is the benchmark name.

        Args:
            name (str): internal name of benchmark.
            class_def (Benchmark): class object of benchmark.
            parameters (str): predefined parameters of benchmark.
            platform (Platform): Platform types like CUDA, ROCM.
        """
        if not name or not isinstance(name, str):
            logger.log_and_raise(
                TypeError,
                'Name of registered benchmark is not string - benchmark: {}, type: {}'.format(name, type(name))
            )

        if not issubclass(class_def, Benchmark):
            logger.log_and_raise(
                TypeError,
                'Registered class is not subclass of Benchmark - benchmark: {}, type: {}'.format(name, type(class_def))
            )

        if name not in cls.benchmarks:
            cls.benchmarks[name] = dict()

        if platform:
            if platform not in Platform:
                platform_list = list(map(str, Platform))
                logger.log_and_raise(
                    TypeError, 'Unknown platform - benchmark: {}, supportted platforms: {}, but got: {}'.format(
                        name, platform_list, platform
                    )
                )
            if platform in cls.benchmarks[name]:
                logger.warning('Duplicate registration - benchmark: {}, platform: {}'.format(name, platform))

            cls.benchmarks[name][platform] = (class_def, parameters)
        else:
            # If not specified the tag, means the benchmark works for all platforms.
            for p in Platform:
                if p in cls.benchmarks[name]:
                    logger.warning('Duplicate registration - benchmark: {}, platform: {}'.format(name, p))

                cls.benchmarks[name][p] = (class_def, parameters)

        cls.__parse_and_check_args(name, class_def, parameters)

    @classmethod
    def __parse_and_check_args(cls, name, class_def, parameters):
        """Parse and check the predefine parameters.

        If ignore_invalid is True, and 'required' arguments are not set when register the benchmark,
        the arguments should be provided by user in config and skip the arguments checking.

        Args:
            name (str): internal name of benchmark.
            class_def (Benchmark): class object of benchmark.
            parameters (str): predefined parameters of benchmark.
        """
        benchmark = class_def(name, parameters)
        benchmark.add_parser_arguments()
        ret, args, unknown = benchmark.parse_args(ignore_invalid=True)
        if not ret or len(unknown) >= 1:
            logger.log_and_raise(
                TypeError,
                'Registered benchmark has invalid arguments - benchmark: {}, parameters: {}'.format(name, parameters)
            )
        elif args is not None:
            cls.benchmarks[name]['predefine_param'] = vars(args)
            logger.debug('Benchmark registration - benchmark: {}, predefine_parameters: {}'.format(name, vars(args)))
        else:
            cls.benchmarks[name]['predefine_param'] = dict()
            logger.info(
                'Benchmark registration - benchmark: {}, missing required parameters or invalid parameters, '
                'skip the arguments checking.'.format(name)
            )

    @classmethod
    def is_benchmark_context_valid(cls, benchmark_context):
        """Check wether the benchmark context is valid or not.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            ret (bool): return True if context is valid.
        """
        if isinstance(benchmark_context, BenchmarkContext) and benchmark_context.name:
            return True
        else:
            logger.error('Benchmark has invalid context')
            return False

    @classmethod
    def __get_benchmark_name(cls, benchmark_context):
        """Return the internal benchmark name.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            benchmark_name (str): internal benchmark name, None means context is invalid.
        """
        if not cls.is_benchmark_context_valid(benchmark_context):
            return None

        benchmark_name = benchmark_context.name
        framework = benchmark_context.framework

        if framework != Framework.NONE:
            benchmark_name = framework.value + '-' + benchmark_name

        return benchmark_name

    @classmethod
    def create_benchmark_context(cls, name, platform=Platform.CPU, parameters='', framework=Framework.NONE):
        """Constructor.

        Args:
            name (str): name of benchmark in config file.
            platform (Platform): Platform types like Platform.CPU, Platform.CUDA, Platform.ROCM.
            parameters (str): predefined parameters of benchmark.
            framework (Framework): Framework types like Framework.PYTORCH, Framework.ONNXRUNTIME.

        Return:
            benchmark_context (BenchmarkContext): the benchmark context.
        """
        return BenchmarkContext(name, platform, parameters, framework)

    @classmethod
    def get_benchmark_configurable_settings(cls, benchmark_context):
        """Get all configurable settings of benchmark.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            All configurable settings in raw string, None means context is invalid or no benchmark is found.
        """
        if not cls.is_benchmark_context_valid(benchmark_context):
            return None

        benchmark_name = cls.__get_benchmark_name(benchmark_context)
        platform = benchmark_context.platform

        (benchmark_class, predefine_params) = cls.__select_benchmark(benchmark_name, platform)
        if benchmark_class:
            benchmark = benchmark_class(benchmark_name)
            benchmark.add_parser_arguments()
            return benchmark.get_configurable_settings()
        else:
            return None

    @classmethod
    def get_all_benchmark_predefine_settings(cls):
        """Get all registered benchmarks' predefine settings.

        Return:
            benchmark_params (dict[str, dict]): key is benchmark name,
              value is the dict with structure: {'parameter': default_value}.
        """
        benchmark_params = dict()
        for name in cls.benchmarks:
            benchmark_params[name] = cls.benchmarks[name]['predefine_param']
        return benchmark_params

    @classmethod
    def launch_benchmark(cls, benchmark_context):
        """Select and Launch benchmark.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            benchmark (Benchmark): the benchmark instance contains all results,
              None means context is invalid or no benchmark is found.
        """
        if not cls.is_benchmark_context_valid(benchmark_context):
            return None

        benchmark_name = cls.__get_benchmark_name(benchmark_context)

        benchmark = None
        if benchmark_name:
            platform = benchmark_context.platform
            parameters = benchmark_context.parameters
            (benchmark_class, predefine_params) = cls.__select_benchmark(benchmark_name, platform)
            if benchmark_class:
                if predefine_params:
                    parameters = predefine_params + ' ' + parameters

                benchmark = benchmark_class(benchmark_name, parameters)
                benchmark.run()

        return benchmark

    @classmethod
    def is_benchmark_registered(cls, benchmark_context):
        """Check wether the benchmark is registered or not.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            ret (bool): return True if context is valid and benchmark is registered.
        """
        if not cls.is_benchmark_context_valid(benchmark_context):
            return False

        benchmark_name = cls.__get_benchmark_name(benchmark_context)
        platform = benchmark_context.platform

        if cls.benchmarks.get(benchmark_name, {}).get(platform) is None:
            return False

        return True

    @classmethod
    def __select_benchmark(cls, name, platform):
        """Select benchmark by name and platform.

        Args:
            name (str): internal name of benchmark.
            platform (Platform): Platform type of benchmark.

        Return:
            benchmark_class (Benchmark): class object of benchmark.
            predefine_params (str): predefined parameters which is set when register the benchmark.
        """
        if name not in cls.benchmarks or platform not in cls.benchmarks[name]:
            logger.warning('Benchmark has no implementation, name: {}, platform: {}'.format(name, platform))
            return (None, None)

        (benchmark_class, predefine_params) = cls.benchmarks[name][platform]

        return (benchmark_class, predefine_params)

    @classmethod
    def clean_benchmarks(cls):
        """Clean up the benchmark registry."""
        cls.benchmarks.clear()
