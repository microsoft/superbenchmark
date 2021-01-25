# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from superbench.common.utils import logger
from superbench.benchmarks.utils import Platform
from superbench.benchmarks.utils import detect_platform


class BenchmarkRegistry:
    benchmarks = dict()

    @staticmethod
    def register_benchmark(name, class_def, parameters=None, tag=None):
        ''' Register new benchmark, key is the benchmark name.

        Args:
            name (str): name of benchmark.
            class_def (Benchmark): class object of benchmark.
            parameters (str): predefined parameters of benchmark.
            tag (Platform): Platform types like cuda, rocm.
        '''
        if name not in BenchmarkRegistry.benchmarks:
            BenchmarkRegistry.benchmarks[name] = dict()
        if not tag:
            for p in Platform:
                if p not in BenchmarkRegistry.benchmarks[name]:
                    BenchmarkRegistry.benchmarks[name][p] = (
                        class_def, parameters)
                else:
                    assert(False), \
                        'Duplicate benchmark registry, name: {}'.format(name)
        else:
            if tag not in Platform:
                assert(False), 'Only support platform tag, got: {}'.format(tag)

            if tag not in BenchmarkRegistry.benchmarks[name]:
                BenchmarkRegistry.benchmarks[name][tag] = (
                    class_def, parameters)
            else:
                assert(False), \
                    'Duplicate benchmark registry, name: {}, tag: {}'.format(
                        name, tag)

    @staticmethod
    def select_benchmark(name, parameters=''):
        ''' Select benchmark by name.

        Args:
            name (str): name of benchmark.
            parameters (str): customized parameters of benchmark.

        Return:
            benchmark_class (Benchmark): class object of benchmark.
            parameters (str): combination of customized and
                              predefined parameters.
        '''
        tag = detect_platform()
        if tag not in BenchmarkRegistry.benchmarks[name]:
            logger.warning(
                'Benchmark has no implementation, name: {}, tag: {}'.format(
                    name, tag))
            return (None, None)

        (benchmark_class,
         predefine_args) = BenchmarkRegistry.benchmarks[name][tag]
        if predefine_args:
            parameters = predefine_args + ' ' + parameters
        return (benchmark_class, parameters)

    @staticmethod
    def get_benchmark_name_and_parameters(config):
        ''' Extract benchmark name and customized parameters from user config.

        Args:
            config (object): the customized benchmark config.

        Return:
            name (str): internal benchmark name.
            parameters (str): customized benchmark parameters.
        '''
        # TODO: add the extraction logic.
        pass

    @staticmethod
    def check_benchmark_parameters(name, parameters):
        ''' Check the validation of benchmark parameters.

        Args:
            name (str): internal benchmark name.
            parameters (str): customized benchmark parameters.

        Return:
            return True if parameters are valid.
        '''
        (benchmark_class, parameters) = BenchmarkRegistry.select_benchmark(
            name, parameters)
        if benchmark_class:
            benchmark = benchmark_class(name, parameters.split(' '))
            # benchmark.add_parser_auguments()
            args, unknown = benchmark.parse_args()
            if len(unknown) > 1:
                return False
            return True
        else:
            return False

    @staticmethod
    def get_benchmark_configurable_settings(name):
        ''' Get all the configurable settings of benchmark.

        Args:
            name (str): internal benchmark name.

        Return:
            all configurable settings in raw string.
        '''
        (benchmark_class, predefine_args) = \
            BenchmarkRegistry.select_benchmark(name)
        if benchmark_class:
            return benchmark_class(name).get_configurable_settings()
        else:
            return None

    @staticmethod
    def launch_benchmark(name, parameters):
        ''' Select and Launch benchmark by name.

        Args:
            name (str): name of benchmark.
            parameters (str): customized parameters of benchmark.

        Return:
            serialized result string with json format.
        '''
        result = None
        if BenchmarkRegistry.check_benchmark_parameters(name, parameters):
            (benchmark_class, parameters) = BenchmarkRegistry.select_benchmark(
                name, parameters)
            if benchmark_class:
                benchmark = benchmark_class(name, parameters.split(' '))
                result = benchmark.run()
        return result
