# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for sharding-matmul with pytorch.

Commands to run:
  python3 -m torch.distributed.launch --nproc_per_node=8 examples/benchmarks/sharding_matmul.py
"""

from superbench.benchmarks import Framework, BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'sharding-matmul', parameters='--num_steps 20', framework=Framework.PYTORCH
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
