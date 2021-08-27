# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for GPU SM copy bandwidth performance.

Commands to run:
  python3 examples/benchmarks/gpu_sm_copy_bw_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'gpu-sm-copy-bw', platform=Platform.CUDA, parameters='--numa_nodes 0 --gpu_ids 0 --enable_dtoh --enable_htod'
    )
    # For ROCm environment, please specify the benchmark name and the platform as the following.
    # context = BenchmarkRegistry.create_benchmark_context(
    #     'gpu-sm-copy-bw', platform=Platform.ROCM, parameters='--numa_nodes 0 --gpu_ids 0 --enable_dtoh --enable_htod'
    # )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
