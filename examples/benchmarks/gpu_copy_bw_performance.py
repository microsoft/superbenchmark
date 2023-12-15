# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for GPU copy bandwidth performance.

Commands to run:
  python3 examples/benchmarks/gpu_copy_bw_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'gpu-copy-bw',
        platform=Platform.CUDA,
        parameters='--mem_type htod dtoh dtod one_to_all all_to_one all_to_all --copy_type sm dma'
    )
    # For ROCm environment, please specify the benchmark name and the platform as the following.
    # context = BenchmarkRegistry.create_benchmark_context(
    #     'gpu-copy-bw', platform=Platform.ROCM, parameters='--mem_type htod dtoh dtod --copy_type sm dma'
    # )
    # For bidirectional test, please specify parameters as the following.
    # parameters='--mem_type htod dtod --copy_type sm dma --bidirectional'
    # To enable data checking, please add '--check_data'.

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
