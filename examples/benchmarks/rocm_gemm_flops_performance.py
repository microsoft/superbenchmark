# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for XDLOPS GEMM FLOPs performance.

Commands to run:
  python3 examples/benchmarks/rocm_gemm_flops_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('gemm-flops', platform=Platform.ROCM)

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
