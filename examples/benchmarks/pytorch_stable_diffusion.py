# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for Stable Diffusion XL models.

Commands to run:
  python3 examples/benchmarks/pytorch_stable_diffusion.py (Single GPU)
  python3 -m torch.distributed.launch --use_env --nproc_per_node=8 examples/benchmarks/pytorch_stable_diffusion.py \
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
    model_name = 'stable-diffusion-xl'
    parameters = '--batch_size 1 --precision float32 float16 --num_warmup 8 --num_steps 64 \
        --sample_count 512 --height 1024 --width 1024 --pin_memory'

    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'

    # Create context for stable diffusion xl benchmark and run it.
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
