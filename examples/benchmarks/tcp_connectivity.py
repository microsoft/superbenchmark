# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Microbenchmark benchmark example for TCP connectivity.

Commands to run:
  python3 examples/benchmarks/tcp_connectivity.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'tcp-connectivity', parameters='--hostfile /tmp/superbench/hostfile.test --port 80 --parallel 1'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
