# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Docker benchmark example for FAMBench.

Commands to run:
  python3 examples/benchmarks/fambench.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('fambench', platform=Platform.CUDA)

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
