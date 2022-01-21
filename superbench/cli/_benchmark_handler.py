# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI benchmark subgroup command handler."""

import re
from pprint import pformat

from knack.util import CLIError

from superbench.benchmarks import Platform, BenchmarkRegistry


def benchmark_list_command_handler(name=None):
    """List benchmarks which match the regular expression.

    Args:
        name (str, optional): Benchmark name or regular expression. Defaults to None.

    Raises:
        CLIError: If cannot find the matching benchmark.

    Returns:
        list: Benchmark list.
    """
    benchmark_list = list(BenchmarkRegistry.get_all_benchmark_predefine_settings().keys())
    if name is None:
        return benchmark_list
    filter_list = list(filter(re.compile(name).match, benchmark_list))
    if not filter_list:
        raise CLIError('Benchmark {} does not exist.'.format(name))
    return filter_list


def benchmark_list_params_command_handler(name=None):
    """List parameters for benchmarks which match the regular expression.

    Args:
        name (str, optional): Benchmark name or regular expression. Defaults to None.

    Raises:
        CLIError: If cannot find the matching benchmark.
    """
    for benchmark_name in benchmark_list_command_handler(name):
        format_help = ''
        for platform in Platform:
            if platform in BenchmarkRegistry.benchmarks[benchmark_name]:
                format_help = BenchmarkRegistry.get_benchmark_configurable_settings(
                    BenchmarkRegistry.create_benchmark_context(benchmark_name, platform=platform)
                )
                break
        print(
            (
                f'=== {benchmark_name} ===\n\n'
                f'{format_help}\n\n'
                f'default values:\n'
                f'{pformat(BenchmarkRegistry.benchmarks[benchmark_name]["predefine_param"])}\n'
            )
        )
