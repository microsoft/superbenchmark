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
        'cpu-stream', parameters='--cpu_arch neo2 '
        '--numa_mem_nodes 0 '
        '--cores 0 1 2 3'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
