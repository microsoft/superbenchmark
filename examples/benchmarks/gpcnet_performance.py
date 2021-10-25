# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for gpcnet performance.

Commands to run:
  mpirun --allow-run-as-root -np 2 -H node0:1,node1:1 examples/benchmarks/gpcnet_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('gpcnet-network-test')
    # context = BenchmarkRegistry.create_benchmark_context('gpcnet-network-load-test')

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
