# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from superbench.benchmarks.utils.platform_detector import Platform, detect_platform


class BenchmarkRegistry:
    benchmarks = dict()

    @staticmethod
    def register_benchmark(name, class_def, parameters=None, tag=None):
        """
        Args:
            name (str): name of benchmark.
            class_def (Benchmark): class of benchmark.
            parameters (str): parameters of benchmark.
            tag (Platform): Platform types like cuda, rocm.
        """

        if name not in BenchmarkRegistry.benchmarks:
            BenchmarkRegistry.benchmarks[name] = dict()
        if not tag:
            for p in Platform:
                if p not in BenchmarkRegistry.benchmarks[name]:
                    BenchmarkRegistry.benchmarks[name][p] = (
                        class_def, parameters)
                else:
                    assert(False), 'Duplicate benchmark registry, name: {}'.format(
                        name)
        else:
            if tag not in Platform:
                assert(False), 'Only support platform tag, got: {}'.format(tag)

            if tag not in BenchmarkRegistry.benchmarks[name]:
                BenchmarkRegistry.benchmarks[name][tag] = (
                    class_def, parameters)
            else:
                assert(False), 'Duplicate benchmark registry, name: {}, tag: {}'.format(
                    name, tag)

    @staticmethod
    def select_benchmark(name, parameters=''):
        tag = detect_platform()
        if tag not in BenchmarkRegistry.benchmarks[name]:
            print("Benchmark has no implementation, name: {}, tag: {}".format(name, tag))
            return (None, None)

        (benchmark_class,
         predefine_args) = BenchmarkRegistry.benchmarks[name][tag]
        if predefine_args:
            parameters = predefine_args + ' ' + parameters
        return (benchmark_class, parameters)

    @staticmethod
    def get_benchmark_name_and_parameters(config):
        """
        Args:
            config (object): the benchmark config want to launch.

        Returns:
            name (str): internal benchmark name.
            parameters (str): customized benchmark parameters.
        """
        pass

    @staticmethod
    def check_benchmark_parameters(name, parameters):
        """
        Args:
            name (str): internal benchmark name.
            parameters (str): customized benchmark parameters.

        Returns:
            return true if the customized parameters is valid.
        """
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
        """
        Args:
            name (str): internal benchmark name.

        Returns:
            settings (str): all configurable settings in raw string.
        """
        (benchmark_class, predefine_args) = BenchmarkRegistry.select_benchmark(name)
        if benchmark_class:
            return benchmark_class(name).get_configurable_settings()
        else:
            return None

    @staticmethod
    def launch_benchmark(name, parameters):
        """
        Args:
            name (str): name of benchmark.
            parameters (str): parameters of benchmark.

        Returns:
            result object with json format.
        """
        result = None
        if BenchmarkRegistry.check_benchmark_parameters(name, parameters):
            (benchmark_class, parameters) = BenchmarkRegistry.select_benchmark(
                name, parameters)
            if benchmark_class:
                benchmark = benchmark_class(name, parameters.split(' '))
                result = benchmark.run()
        return result
