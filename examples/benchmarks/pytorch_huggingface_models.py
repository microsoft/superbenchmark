# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for loading models from HuggingFace Hub.

This example demonstrates how to benchmark models loaded directly from
HuggingFace Hub instead of using in-house model implementations.

Commands to run:
  python3 examples/benchmarks/pytorch_huggingface_models.py (Single GPU)
  python3 examples/benchmarks/pytorch_huggingface_models.py --model bert (BERT model)
  python3 examples/benchmarks/pytorch_huggingface_models.py --model gpt2 (GPT-2 model)
  torchrun --nproc_per_node=2 examples/benchmarks/pytorch_huggingface_models.py --distributed (Distributed, 2 GPUs)
  torchrun --nproc_per_node=$NUM_GPUS examples/benchmarks/pytorch_huggingface_models.py --distributed (Distributed, N GPUs)

Environment variables:
  HF_TOKEN: HuggingFace token for gated models (optional)
"""

import argparse
import os

from superbench.benchmarks import Platform, Framework, BenchmarkRegistry
from superbench.common.utils import logger

# Define HuggingFace models to benchmark
HF_MODELS = {
    'bert': {
        'name': 'bert-base',
        'identifier': 'google-bert/bert-base-uncased',
        'parameters': '--batch_size 32 --seq_len 128 --num_classes 2',
    },
    'gpt2': {
        'name': 'gpt2-small',
        'identifier': 'openai-community/gpt2',
        'parameters': '--batch_size 8 --seq_len 128',
    },
}


def run_huggingface_benchmark(model_key, distributed=False, precision='float32', duration=60):
    """Run a benchmark using a HuggingFace model.
    
    Args:
        model_key: Key to look up model config in HF_MODELS.
        distributed: Whether to enable distributed training.
        precision: Model precision (float32, float16).
        duration: Benchmark duration in seconds.
    """
    if model_key not in HF_MODELS:
        logger.error(f'Unknown model: {model_key}. Available: {list(HF_MODELS.keys())}')
        return None
    
    model_config = HF_MODELS[model_key]
    model_name = model_config['name']
    hf_identifier = model_config['identifier']
    
    # Build parameters with HuggingFace model source
    parameters = (
        f"{model_config['parameters']} "
        f"--duration {duration} "
        f"--precision {precision} "
        f"--run_count 2 "
        f"--model_source huggingface "
        f"--model_identifier {hf_identifier}"
    )
    
    if distributed:
        parameters += ' --distributed_impl ddp --distributed_backend nccl'
    
    logger.info(f'Running HuggingFace benchmark: {model_name} ({hf_identifier})')
    logger.info(f'Parameters: {parameters}')
    
    # Create context and run benchmark
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
    
    return benchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark HuggingFace models with SuperBench'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='bert',
        choices=list(HF_MODELS.keys()),
        help='Model to benchmark (default: bert)'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        default=False,
        help='Whether to enable distributed training.'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='float32',
        choices=['float32', 'float16'],
        help='Model precision (default: float32)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Benchmark duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        default=False,
        help='Run benchmarks for all available models'
    )
    args = parser.parse_args()
    
    if args.all:
        # Run all models
        for model_key in HF_MODELS:
            run_huggingface_benchmark(
                model_key,
                distributed=args.distributed,
                precision=args.precision,
                duration=args.duration
            )
    else:
        # Run single model
        run_huggingface_benchmark(
            args.model,
            distributed=args.distributed,
            precision=args.precision,
            duration=args.duration
        )
