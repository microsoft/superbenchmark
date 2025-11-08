# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Example of NVBench Sleep Kernel benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger


def main():
    """Main method to run the nvbench sleep kernel benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'nvbench-sleep-kernel', platform=Platform.CUDA, parameters='--duration_us "[25,50,75]" --timeout 10'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
    else:
        logger.error('benchmark: nvbench-sleep-kernel launch failed.')


if __name__ == '__main__':
    main()
