# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for device memory bandwidth performance.

Commands to run:
  python3 examples/benchmarks/rocm_memory_bw_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('mem-bw', platform=Platform.ROCM)

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
