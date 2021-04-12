# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for matmul with pytorch.

Commands to run:
  python3 examples/benchmarks/matmul.py
"""

from superbench.benchmarks import Framework, BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'matmul', parameters='--num_steps 20', framework=Framework.PYTORCH
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
