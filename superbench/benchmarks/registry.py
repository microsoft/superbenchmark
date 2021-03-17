# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Interfaces that provide access to benchmarks."""

from typing import Dict

from superbench.common.utils import logger
from superbench.common.errors import DuplicateBenchmarkRegistrationError
from superbench.benchmarks import Platform, Framework, BenchmarkContext
from superbench.benchmarks.base import Benchmark
import superbench.benchmarks.model_benchmarks    # noqa # pylint: disable=unused-import
import superbench.benchmarks.micro_benchmarks    # noqa # pylint: disable=unused-import
import superbench.benchmarks.docker_benchmarks    # noqa # pylint: disable=unused-import


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
    def register_benchmark(cls, name, class_def, parameters=None, platform=None):
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

            if platform not in cls.benchmarks[name]:
                cls.benchmarks[name][platform] = (class_def, parameters)
            else:
                logger.log_and_raise(
                    DuplicateBenchmarkRegistrationError,
                    'Duplicate registration - benchmark: {}, platform: {}'.format(name, platform)
                )
        else:
            # If not specified the tag, means the
            # benchmark works for all platforms.
            for p in Platform:
                if p not in cls.benchmarks[name]:
                    cls.benchmarks[name][p] = (class_def, parameters)
                else:
                    logger.log_and_raise(
                        DuplicateBenchmarkRegistrationError, 'Duplicate registration - benchmark: {}'.format(name)
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
    def check_parameters(cls, benchmark_context):
        """Check the validation of customized parameters.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            Return True if benchmark exists and context/parameters are valid.
        """
        if not cls.is_benchmark_context_valid(benchmark_context):
            return False

        benchmark_name = cls.__get_benchmark_name(benchmark_context)
        platform = benchmark_context.platform
        customized_parameters = benchmark_context.parameters

        if benchmark_name:
            (benchmark_class, params) = cls.__select_benchmark(benchmark_name, platform)
            if benchmark_class:
                benchmark = benchmark_class(benchmark_name, customized_parameters)
                benchmark.add_parser_arguments()
                ret, args, unknown = benchmark.parse_args()
                if ret and len(unknown) < 1:
                    return True

        return False

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
