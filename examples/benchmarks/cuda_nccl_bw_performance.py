# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for NCCL Bus bandwidth performance.

Commands to run:
  python3 examples/benchmarks/nccl_bw_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'nccl-bw', platform=Platform.CUDA, parameters='--operations allreduce'
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
