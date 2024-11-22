# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for nvbandwidth benchmark.

Commands to run:
  python3 examples/benchmarks/nvbandwidth.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'nvbandwidth',
        platform=Platform.CPU,
        parameters=(
            '--buffer_size 128 '
            '--test_cases 0,1,19,20 '
            '--skip_verification '
            '--disable_affinity '
            '--use_mean '
            '--num_loops 10'
        )
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
