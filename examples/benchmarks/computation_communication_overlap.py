# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for computation and communication overlap with pytorch.

Commands to run:
  python3 -m torch.distributed.launch --nproc_per_node=8 examples/benchmarks/computation_communication_overlap.py
"""

from superbench.benchmarks import Framework, BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'computation-communication-overlap', parameters='--num_steps 2000', framework=Framework.PYTORCH
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
