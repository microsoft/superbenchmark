# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for kernel launch overhead.

Commands to run:
  python3 examples/benchmarks/kernel_launch_overhead.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('kernel-launch')

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
