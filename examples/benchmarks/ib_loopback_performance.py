# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for IB loopback performance.

Commands to run:
  python examples/benchmarks/ib_loopback_performance_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('ib-loopback')

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
