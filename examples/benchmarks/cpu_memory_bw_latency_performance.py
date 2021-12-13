# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for disk performance.

Commands to run:
  python3 examples/benchmarks/memory_bw_latency_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'cpu-memory-bw-latency',
        platform=Platform.CPU,
        parameters='--tests bandwidth_matrix latency_matrix max_bandwidth'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
