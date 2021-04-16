# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for resnet101.

Commands to run:
  python3 examples/benchmarks/pytorch_cnn_resnet101.py (Single GPU)
  python3 -m torch.distributed.launch --use_env --nproc_per_node=8 examples/benchmarks/pytorch_cnn_resnet101.py \
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
    model_name = 'resnet101'
    parameters = '--batch_size 1 --image_size 224 --precision float32 --num_warmup 8 --num_steps 64 --run_count 2'
    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'

    # Create context for resnet101 benchmark and run it for 64 steps.
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
