# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for cublas performance benchmark.

Commands to run:
  python3 examples/benchmarks/cublas_function.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    parameters = '--num_warmup 8 --num_steps 100 --num_in_step 1000'
    context = BenchmarkRegistry.create_benchmark_context(
        'cublas-function', platform=Platform.CUDA, parameters=parameters
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
