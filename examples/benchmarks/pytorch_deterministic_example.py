# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unified PyTorch deterministic training example for all supported models.

Commands to run:
Generate log:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 examples/benchmarks/pytorch_deterministic_example.py
--model <model_from_MODEL_CHOICES> --generate-log ./outputs/determinism_ref.json

Compare log:

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 examples/benchmarks/pytorch_deterministic_example.py
--model <model_from_MODEL_CHOICES> --compare-log ./outputs/determinism_ref.json

"""

import argparse
from superbench.benchmarks import BenchmarkRegistry, Framework
from superbench.common.utils import logger

MODEL_CHOICES = [
    'bert-large',
    'gpt2-small',
    'llama2-7b',
    'mixtral-8x7b',
    'resnet101',
    'lstm',
]

DEFAULT_PARAMS = {
    'bert-large':
    '--batch_size 1 --seq_len 64 --num_warmup 1 --num_steps 200 --precision float32 '
    '--model_action train --deterministic --deterministic_seed 42 --check_frequency 20',
    'gpt2-small':
    '--batch_size 1 --num_steps 300 --num_warmup 1 --seq_len 128 --precision float32 '
    '--model_action train --deterministic --deterministic_seed 42 --check_frequency 20',
    'llama2-7b':
    '--batch_size 1 --num_steps 300 --num_warmup 1 --seq_len 512 --precision float32 --model_action train '
    '--deterministic --deterministic_seed 42 --check_frequency 20',
    'mixtral-8x7b':
    '--hidden_size=4096 --num_hidden_layers=32 --num_attention_heads=32 --intermediate_size=14336 '
    '--num_key_value_heads=8 --max_position_embeddings=32768 --router_aux_loss_coef=0.02 '
    '--deterministic --deterministic_seed 42 --check_frequency 20',
    'resnet101':
    '--batch_size 1 --precision float32 --num_warmup 1 --num_steps 120 --sample_count 8192 '
    '--pin_memory --model_action train --deterministic --deterministic_seed 42 --check_frequency 20',
    'lstm':
    '--batch_size 1 --num_steps 100 --num_warmup 1 --seq_len 64 --precision float16 '
    '--model_action train --deterministic --deterministic_seed 42 --check_frequency 20',
}


def main():
    """Main function for determinism example file."""
    parser = argparse.ArgumentParser(description='Unified PyTorch deterministic training example.')
    parser.add_argument('--model', type=str, choices=MODEL_CHOICES, required=True, help='Model to run.')
    parser.add_argument(
        '--generate-log',
        nargs='?',
        const=True,
        default=None,
        help='Enable fingerprint log generation. Optionally specify a path to save the log.',
    )
    parser.add_argument(
        '--compare-log',
        type=str,
        default=None,
        help='Path to reference fingerprint log for comparison.',
    )
    parser.add_argument(
        '--deterministic-seed',
        type=int,
        default=42,
        help='Seed for deterministic training.',
    )
    args = parser.parse_args()

    parameters = DEFAULT_PARAMS[args.model]
    if args.deterministic_seed:
        parameters += f' --deterministic_seed {args.deterministic_seed}'
    if args.generate_log:
        parameters += ' --generate-log'
        if isinstance(args.generate_log, str):
            parameters += f' {args.generate_log}'
    if args.compare_log:
        parameters += f' --compare-log {args.compare_log}'

    context = BenchmarkRegistry.create_benchmark_context(args.model, parameters=parameters, framework=Framework.PYTORCH)
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    logger.info(f'Benchmark finished. Return code: {benchmark.return_code}')
    if hasattr(benchmark, '_model_run_metadata'):
        logger.info(f'Run metadata: {benchmark._model_run_metadata}')
    if hasattr(benchmark, '_model_run_losses'):
        logger.info(f'Losses: {benchmark._model_run_losses[:5]} ...')
    if hasattr(benchmark, '_model_run_periodic'):
        logger.info(f'Periodic: {benchmark._model_run_periodic}')


if __name__ == '__main__':
    main()
