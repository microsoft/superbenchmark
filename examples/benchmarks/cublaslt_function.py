# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for cuBLASLt GEMM performance benchmark.

Commands to run:
  python3 examples/benchmarks/cublaslt_function.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    # Basic usage without autotune
    print('Running cuBLASLt benchmark without autotune...')
    parameters = '--num_warmup 10 --num_steps 50 --shapes 512,512,512 --in_types fp16 fp32'
    context = BenchmarkRegistry.create_benchmark_context('cublaslt-gemm', platform=Platform.CUDA, parameters=parameters)

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )

    # Enhanced usage with autotune enabled
    print('\nRunning cuBLASLt benchmark with autotune enabled...')
    parameters_autotune = (
        '--num_warmup 10 --num_steps 50 '
        '--shapes 512,512,512 1024,1024,1024 --in_types fp16 fp32 '
        '--enable_autotune --num_warmup_autotune 20 --num_steps_autotune 50'
    )
    context_autotune = BenchmarkRegistry.create_benchmark_context(
        'cublaslt-gemm', platform=Platform.CUDA, parameters=parameters_autotune
    )

    benchmark_autotune = BenchmarkRegistry.launch_benchmark(context_autotune)
    if benchmark_autotune:
        logger.info(
            'benchmark with autotune: {}, return code: {}, result: {}'.format(
                benchmark_autotune.name, benchmark_autotune.return_code, benchmark_autotune.result
            )
        )

    # FP8 specific usage with autotune
    print('\nRunning cuBLASLt benchmark with FP8 and autotune...')
    parameters_fp8 = (
        '--num_warmup 5 --num_steps 20 '
        '--shapes 512,512,512 --in_types fp8e4m3 fp8e5m2 '
        '--enable_autotune --num_warmup_autotune 10 --num_steps_autotune 30'
    )
    context_fp8 = BenchmarkRegistry.create_benchmark_context(
        'cublaslt-gemm', platform=Platform.CUDA, parameters=parameters_fp8
    )

    benchmark_fp8 = BenchmarkRegistry.launch_benchmark(context_fp8)
    if benchmark_fp8:
        logger.info(
            'FP8 benchmark with autotune: {}, return code: {}, result: {}'.format(
                benchmark_fp8.name, benchmark_fp8.return_code, benchmark_fp8.result
            )
        )
