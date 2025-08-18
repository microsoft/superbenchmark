# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for CNN models.

Commands to run:
  python3 examples/benchmarks/pytorch_cnn.py (Single GPU)
  python3 -m torch.distributed.launch --use_env --nproc_per_node=8 examples/benchmarks/pytorch_cnn.py \
      --distributed (Distributed)

Deterministic + logging:
    # Generate reference log (determinism). Requires cuBLAS env.
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 examples/benchmarks/pytorch_cnn.py \
            --deterministic --random_seed 42 --generate_log --log_path ./outputs/cnn_ref.json \
            --check_frequency 50

    # Compare against reference
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 examples/benchmarks/pytorch_cnn.py \
            --deterministic --random_seed 42 --compare_log ./outputs/cnn_ref.json \
            --check_frequency 50
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
    parser.add_argument('--check_frequency', type=int, default=None, help='Step cadence for periodic checks/logging.')
    # Logging / comparison
    parser.add_argument('--generate_log', action='store_true', default=False, help='Save fingerprint log to file.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to save or load fingerprint log.')
    parser.add_argument('--compare_log', type=str, default=None, help='Compare this run to a reference fingerprint log.')
    args = parser.parse_args()

    # Specify the model name and benchmark parameters.
    # For example, resnet50, resnet101, resnet152, densenet169, densenet201, vgg11, vgg13, vgg16, vgg19.
    model_name = 'resnet101'
    parameters = '--batch_size 192 --precision float32 float16 --num_warmup 64 --num_steps 512 \
        --sample_count 8192 --pin_memory'

    if args.distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'
    if args.deterministic:
        parameters += ' --deterministic --precision float32'
    if args.random_seed is not None:
        parameters += f' --random_seed {args.random_seed}'
    if args.check_frequency is not None:
        parameters += f' --check_frequency {args.check_frequency}'
    if args.generate_log:
        logger.info('Log generation enabled')
        parameters += ' --generate-log'
        if args.log_path:
            parameters += f' --log-path {args.log_path}'
    if args.compare_log:
        parameters += f' --compare-log {args.compare_log}'

    if args.deterministic:
        logger.info('Deterministic run. Ensure CUBLAS_WORKSPACE_CONFIG is set before CUDA init (e.g., :4096:8).')

    # Create context for resnet101 benchmark and run it for 2048 steps.
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
