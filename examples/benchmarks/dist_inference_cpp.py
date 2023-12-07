# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for distributed inference (cpp implementation).

Commands to run:
  mpirun -np 8 examples/benchmarks/dist_inference_cpp.py
"""

from superbench.benchmarks import Framework, BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'dist-inference-cpp', platform=Platform.CUDA, parameters=''
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
