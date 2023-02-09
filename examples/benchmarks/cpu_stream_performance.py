# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for CPU Stream performance.

Commands to run:
  python3 examples/benchmarks/cpu_stream_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'cpu-stream',
        parameters='--cpu_arch zen3 \
        --cores 0 4 8 12 16 20 24 28 30 34 38 42 46 50 54 58 60 64 68 72 76 80 84 88 90 94 98 102 106 110 114 118'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
