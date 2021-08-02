# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for NCCL/RCCL Bus bandwidth performance.

Commands to run:
  python3 examples/benchmarks/nccl_bw_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'nccl-bw', platform=Platform.CUDA, parameters='--operations allreduce'
    )
    # For ROCM environment, please specify the benchmark name and the platform as the following.
    # context = BenchmarkRegistry.create_benchmark_context(
    #     'rccl-bw', platform=Platform.ROCM, parameters='--operations allreduce --maxbytes 128M'
    # )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
