# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for loading models from HuggingFace Hub.

This example demonstrates how to benchmark models loaded directly from
HuggingFace Hub instead of using in-house model implementations.

Commands to run:
  python3 examples/benchmarks/pytorch_huggingface_models.py (Single GPU)
  python3 examples/benchmarks/pytorch_huggingface_models.py --model bert (BERT model)
  python3 examples/benchmarks/pytorch_huggingface_models.py --model gpt2 (GPT-2 model)
  python3 examples/benchmarks/pytorch_huggingface_models.py --model llama2-7b (Llama 2 7B, gated)
  torchrun --nproc_per_node=2 examples/benchmarks/pytorch_huggingface_models.py --distributed (Distributed, 2 GPUs)

Environment variables:
  HF_TOKEN: HuggingFace token for gated models such as Llama (required for meta-llama/*)
"""

import argparse

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
    # Generic GPT/Llama benchmarks served by PytorchHFLanguageModel. These pull the
    # checkpoint straight from the HuggingFace Hub and are gated by the pre-flight
    # memory check, so a model only runs if the hardware can support it.
    'gptj-6b': {
        'name': 'hf-gptj-6b',
        'identifier': 'EleutherAI/gpt-j-6b',
        'parameters': '--batch_size 4 --seq_len 128',
    },
    'gpt-neox-20b': {
        'name': 'hf-gpt-neox-20b',
        'identifier': 'EleutherAI/gpt-neox-20b',
        'parameters': '--batch_size 1 --seq_len 128',
    },
    'llama2-7b': {
        'name': 'hf-llama2-7b',
        'identifier': 'meta-llama/Llama-2-7b-hf',
        'parameters': '--batch_size 4 --seq_len 128',
    },
    'llama2-13b': {
        'name': 'hf-llama2-13b',
        'identifier': 'meta-llama/Llama-2-13b-hf',
        'parameters': '--batch_size 2 --seq_len 128',
    },
    'llama2-70b': {
        'name': 'hf-llama2-70b',
        'identifier': 'meta-llama/Llama-2-70b-hf',
        'parameters': '--batch_size 1 --seq_len 128 --model_action inference',
    },
    'llama3-8b': {
        'name': 'hf-llama3-8b',
        'identifier': 'meta-llama/Meta-Llama-3-8B',
        'parameters': '--batch_size 4 --seq_len 128',
    },
    'llama3.2-1b': {
        'name': 'hf-llama3.2-1b',
        'identifier': 'meta-llama/Llama-3.2-1B',
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
        f'--duration {duration} '
        f'--precision {precision} '
        f'--run_count 2 '
        f'--model_source huggingface '
        f'--model_identifier {hf_identifier}'
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
    parser = argparse.ArgumentParser(description='Benchmark HuggingFace models with SuperBench')
    parser.add_argument(
        '--model', type=str, default='bert', choices=list(HF_MODELS.keys()), help='Model to benchmark (default: bert)'
    )
    parser.add_argument(
        '--distributed', action='store_true', default=False, help='Whether to enable distributed training.'
    )
    parser.add_argument(
        '--precision',
        type=str,
        default='float32',
        choices=['float32', 'float16'],
        help='Model precision (default: float32)'
    )
    parser.add_argument('--duration', type=int, default=60, help='Benchmark duration in seconds (default: 60)')
    parser.add_argument('--all', action='store_true', default=False, help='Run benchmarks for all available models')
    args = parser.parse_args()

    if args.all:
        # Run all models
        for model_key in HF_MODELS:
            run_huggingface_benchmark(
                model_key, distributed=args.distributed, precision=args.precision, duration=args.duration
            )
    else:
        # Run single model
        run_huggingface_benchmark(
            args.model, distributed=args.distributed, precision=args.precision, duration=args.duration
        )
