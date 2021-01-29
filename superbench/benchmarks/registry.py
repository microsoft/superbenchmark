# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Interfaces that provide access to benchmarks."""

from typing import Dict

from superbench.common.utils import logger
from superbench.benchmarks import Platform, Framework


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
    def register_benchmark(cls,
                           name,
                           class_def,
                           parameters=None,
                           platform=None):
        """Register new benchmark, key is the benchmark name.

        Args:
            name (str): internal name of benchmark.
            class_def (Benchmark): class object of benchmark.
            parameters (str): predefined parameters of benchmark.
            platform (Platform): Platform types like CUDA, ROCM.
        """
        if name not in cls.benchmarks:
            cls.benchmarks[name] = dict()

        if platform:
            if platform not in Platform:
                platform_list = list(map(str, Platform))
                assert(False), \
                    'Supportted platforms: {}, but got: {}'.format(
                        platform_list, platform)

            if platform not in cls.benchmarks[name]:
                cls.benchmarks[name][platform] = (class_def, parameters)
            else:
                assert(False), \
                    'Duplicate registration, name: {}, platform: {}'.format(
                        name, platform)
        else:
            # If not specified the tag, means the
            # benchmark works for all platforms.
            for p in Platform:
                if p not in cls.benchmarks[name]:
                    cls.benchmarks[name][p] = (class_def, parameters)
                else:
                    assert(False), \
                        'Duplicate registration, name: {}'.format(name)

    @classmethod
    def get_benchmark_name(cls, benchmark_context):
        """Return the internal benchmark name.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            benchmark_name (str): internal benchmark name,
              None means no benchmark found.
        """
        benchmark_name = benchmark_context.name
        platform = benchmark_context.platform
        framework = benchmark_context.framework

        if framework != Framework.NONE:
            benchmark_name = framework.value + '-' + benchmark_name

        if not cls._is_benchmark_registered(benchmark_name, platform):
            benchmark_name = None

        return benchmark_name

    @classmethod
    def check_parameters(cls, benchmark_context):
        """Check the validation of customized parameters.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            ret (Boolean): return True if parameters are valid.
        """
        benchmark_name = cls.get_benchmark_name(benchmark_context)
        platform = benchmark_context.platform
        customized_parameters = benchmark_context.parameters

        ret = False
        if benchmark_name:
            (benchmark_class,
             params) = cls._select_benchmark(benchmark_name, platform)
            benchmark = benchmark_class(benchmark_name, customized_parameters)
            benchmark.add_parser_auguments()
            args, unknown = benchmark.parse_args()
            if len(unknown) < 1:
                ret = True

        return ret

    @classmethod
    def get_benchmark_configurable_settings(cls, name, platform):
        """Get all configurable settings of benchmark.

        Args:
            name (str): internal benchmark name got from
                        get_benchmark_name().
            platform (Platform): Platform type of benchmark.

        Return:
            All configurable settings in raw string, None means
            no benchmark found.
        """
        (benchmark_class, predefine_params) = \
            cls._select_benchmark(name, platform)
        if benchmark_class:
            benchmark = benchmark_class(name)
            benchmark.add_parser_auguments()
            return benchmark.get_configurable_settings()
        else:
            return None

    @classmethod
    def launch_benchmark(cls, benchmark_context):
        """Select and Launch benchmark.

        Args:
            benchmark_context (BenchmarkContext): the benchmark context.

        Return:
            Serialized result string with json format, None means
            no benchmark found.
        """
        benchmark_name = cls.get_benchmark_name(benchmark_context)

        result = None
        if benchmark_name:
            platform = benchmark_context.platform
            parameters = benchmark_context.parameters
            (benchmark_class, predefine_params) = \
                cls._select_benchmark(benchmark_name, platform)
            if predefine_params:
                parameters = predefine_params + ' ' + parameters

            benchmark = benchmark_class(benchmark_name, parameters)
            result = benchmark.run()

        return result

    @classmethod
    def _is_benchmark_registered(cls, name, platform):
        """Check wether the benchmark is registered or not.

        Args:
            name (str): internal name of benchmark.
            platform (Platform): Platform type of benchmark.

        Return:
            ret (Boolean): return True if parameters are valid.
        """
        if name not in cls.benchmarks:
            return False

        if platform not in cls.benchmarks[name]:
            return False

        return True

    @classmethod
    def _select_benchmark(cls, name, platform):
        """Select benchmark by name and platform.

        Args:
            name (str): internal name of benchmark.
            platform (Platform): Platform type of benchmark.

        Return:
            benchmark_class (Benchmark): class object of benchmark.
            predefine_params (str): predefined parameters which is set when
              register the benchmark.
        """
        if not cls._is_benchmark_registered(name, platform):
            logger.warning(
                'Benchmark has no realization, name: {}, platform: {}'.format(
                    name, platform))
            return (None, None)

        (benchmark_class, predefine_params) = cls.benchmarks[name][platform]

        return (benchmark_class, predefine_params)

    @classmethod
    def clean_benchmarks(cls):
        """Clean up the benchmark registry."""
        cls.benchmarks.clear()
