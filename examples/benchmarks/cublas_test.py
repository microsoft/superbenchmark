# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for matmul with pytorch.
Commands to run:
  python3 examples/benchmarks/cublas_test.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('cublas-test', parameters='--warm_up 8 --num_steps 100 --num_in_step 1000',)

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )