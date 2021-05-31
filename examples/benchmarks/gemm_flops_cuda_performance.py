# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for Cutlass GEMM FLOPs performance.

Commands to run:
  python3 examples/benchmarks/gemm_flops_cuda_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    parameters = '--n 16384 --k 16384 --m 16384'
    context = BenchmarkRegistry.create_benchmark_context('gemm-flops', platform=Platform.CUDA, parameters=parameters)

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
