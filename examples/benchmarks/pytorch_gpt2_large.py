# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for gpt2-large (36-layer, 1280-hidden, 20-heads, 774M parameters).

Commands to run:
  python3 examples/benchmarks/pytorch_gpt2_large.py (Single GPU)
  python3 -m torch.distributed.launch --use_env --nproc_per_node=8 examples/benchmarks/pytorch_gpt2_large.py \
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
    model_name = 'gpt2-large'
    parameters = '--batch_size 1 --duration 120 --seq_len 128 --precision float32 --run_count 2'
    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'

    # Create context for gpt2-large benchmark and run it for 120 * 2 seconds.
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
