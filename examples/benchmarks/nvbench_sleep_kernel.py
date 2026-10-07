# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for NVBench Sleep Kernel.

Commands to run:
  python3 examples/benchmarks/nvbench_sleep_kernel.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
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
