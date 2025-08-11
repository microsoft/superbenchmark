# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for Llama2-7b (32-layer, 4096-hidden, 32-heads, 7B parameters).

Commands to run:
  python3 examples/benchmarks/pytorch_lstm.py (Single GPU)
  python3 -m torch.distributed.launch --use_env --nproc_per_node=8 examples/benchmarks/pytorch_lstm.py \
      --distributed (Distributed)

  Deterministic examples:
  # Soft determinism (numeric reproducibility target):
  python3 examples/benchmarks/pytorch_llama2.py --deterministic --random_seed 42

  # Strict determinism (exact reproducibility; requires cuBLAS env):
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 examples/benchmarks/pytorch_llama2.py \
      --deterministic --random_seed 42 --strict_determinism
"""

import argparse
import os

from superbench.benchmarks import Platform, Framework, BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--distributed', action='store_true', default=False, help='Whether to enable distributed training.'
    )
    parser.add_argument('--deterministic', action='store_true', default=False, help='Enable deterministic training.')
    parser.add_argument('--random_seed', type=int, default=None, help='Fixed seed when using --deterministic.')
    parser.add_argument(
        '--strict_determinism', action='store_true', default=False,
        help='Enable strict determinism checks (set SB_STRICT_DETERMINISM=1). Requires CUBLAS_WORKSPACE_CONFIG env.'
    )
    args = parser.parse_args()

    # Specify the model name and benchmark parameters.
    model_name = 'llama2-7b'
    parameters = '--batch_size 1 --duration 120 --seq_len 512 --precision float16'
    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'
    if args.deterministic:
        parameters += ' --deterministic --precision float32'
    if args.random_seed is not None:
        parameters += f' --random_seed {args.random_seed}'

    if args.strict_determinism:
        # Hint: CUBLAS_WORKSPACE_CONFIG must be set by the user before CUDA init for strict reproducibility.
        os.environ['SB_STRICT_DETERMINISM'] = '1'
        logger.info('Strict determinism enabled (SB_STRICT_DETERMINISM=1). Ensure CUBLAS_WORKSPACE_CONFIG is set.')

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
