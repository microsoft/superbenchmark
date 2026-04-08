# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for NVBench Kernel Launch.

Commands to run:
  python3 examples/benchmarks/nvbench_kernel_launch.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'nvbench-kernel-launch',
        platform=Platform.CUDA,
        parameters=(
            '--timeout 30 '
            '--min-samples 10 '
            '--min-time 1.0 '
            '--max-noise 0.1 '
            '--stopping-criterion stdrel '
            '--throttle-threshold 80 '
            '--throttle-recovery-delay 1.0'
        )
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
