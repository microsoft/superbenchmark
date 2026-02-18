# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for NVBench Auto Throughput.

Commands to run:
  python3 examples/benchmarks/nvbench_auto_throughput.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'nvbench-auto-throughput',
        platform=Platform.CUDA,
        parameters='--stride "[1,2,4,8]" --block_size "[256,512]" --timeout 30'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
