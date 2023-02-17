# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for CPU HPL performance.

Commands to run:
  python3 examples/benchmarks/cpu_hpl_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'cpu-hpl',
        parameters='--cpu_arch zen3 \
        --blockSize 224 --coreCount 60 --blocks 1 --problemSize 224000'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
