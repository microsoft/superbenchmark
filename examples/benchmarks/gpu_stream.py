# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for GPU Stream performance.

Commands to run:
  python3 examples/benchmarks/gpu_stream.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'gpu-stream', platform=Platform.CUDA, parameters='--num_warm_up 1 --num_loops 10 --data_type double'
    )
    # For ROCm environment, please specify the benchmark name and the platform as the following.
    # context = BenchmarkRegistry.create_benchmark_context(
    #     'gpu-stream', platform=Platform.ROCM, parameters='--num_warm_up 1 --num_loops 10'
    # )
    # To enable data checking, please add '--check_data'.

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
