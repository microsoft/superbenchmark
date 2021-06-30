# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for memory copy bandwidth performance.

Commands to run:
  python3 examples/benchmarks/memory_copy_bw_cuda_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('mem-copy-bw', platform=Platform.CUDA)

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
