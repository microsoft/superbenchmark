# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for Llama2-7b (32-layer, 4096-hidden, 32-heads, 7B parameters).

Commands to run:
  python3 examples/benchmarks/pytorch_llama2.py (Single GPU)
  python3 -m torch.distributed.launch --use_env --nproc_per_node=8 examples/benchmarks/pytorch_llama2.py \
      --distributed (Distributed)
"""

import argparse

from superbench.benchmarks import Platform, Framework, BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--distributed', action='store_true', default=False, help='Whether to enable distributed training.'
    )
    args = parser.parse_args()

    # Specify the model name and benchmark parameters.
    model_name = 'llama2-7b'
    parameters = '--batch_size 1 --duration 120 --seq_len 512 --precision float16'
    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'

    # Create context for Llama2 benchmark and run it for 120 seconds.
    context = BenchmarkRegistry.create_benchmark_context(
        model_name, platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
