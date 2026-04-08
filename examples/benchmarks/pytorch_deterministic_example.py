# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unified PyTorch deterministic training example for all supported models.

Deterministic metrics (loss, activation mean) are automatically stored in results
when --enable_determinism flag is enabled.

To compare deterministic results between runs, use the `sb result diagnosis` command
with a baseline file and comparison rules. See the SuperBench documentation for details.

Example workflow:
1. Run first benchmark (creates outputs/<timestamp>/results-summary.jsonl):
   python3 examples/benchmarks/pytorch_deterministic_example.py \
       --model resnet101 --enable_determinism --deterministic_seed 42

2. Generate baseline from results:
   sb result generate-baseline --data-file outputs/<timestamp>/results-summary.jsonl \
       --summary-rule-file summary-rules.yaml --output-dir outputs/<timestamp>

3. Run second benchmark:
   python3 examples/benchmarks/pytorch_deterministic_example.py \
       --model resnet101 --enable_determinism --deterministic_seed 42

4. Compare runs with diagnosis:
   sb result diagnosis --data-file outputs/<run2-timestamp>/results-summary.jsonl \
       --rule-file rules.yaml --baseline-file outputs/<run1-timestamp>/baseline.json

Note: CUBLAS_WORKSPACE_CONFIG is now automatically set by the code when determinism is enabled.
"""

import argparse
import json
import socket
from datetime import datetime
from pathlib import Path
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
    '--model_action train --check_frequency 20',
    'gpt2-small':
    '--batch_size 1 --num_steps 300 --num_warmup 1 --seq_len 128 --precision float32 '
    '--model_action train --check_frequency 20',
    'llama2-7b':
    '--batch_size 1 --num_steps 300 --num_warmup 1 --seq_len 512 --precision float32 --model_action train '
    '--check_frequency 20',
    'mixtral-8x7b':
    '--hidden_size 4096 --num_hidden_layers 32 --num_attention_heads 32 --intermediate_size 14336 '
    '--num_key_value_heads 8 --max_position_embeddings 32768 --router_aux_loss_coef 0.02 '
    '--check_frequency 20',
    'resnet101':
    '--batch_size 1 --precision float32 --num_warmup 1 --num_steps 120 --sample_count 8192 '
    '--pin_memory --model_action train --check_frequency 20',
    'lstm':
    '--batch_size 1 --num_steps 100 --num_warmup 2 --seq_len 64 --precision float32 '
    '--model_action train --check_frequency 30',
}


def main():
    """Main function for determinism example file."""
    parser = argparse.ArgumentParser(description='Unified PyTorch deterministic training example.')
    parser.add_argument('--model', type=str, choices=MODEL_CHOICES, required=True, help='Model to run.')
    parser.add_argument(
        '--enable_determinism',
        action='store_true',
        help='Enable deterministic mode for reproducible results.',
    )
    parser.add_argument(
        '--deterministic_seed',
        type=int,
        default=None,
        help='Seed for deterministic training.',
    )
    args = parser.parse_args()

    parameters = DEFAULT_PARAMS[args.model]
    if args.enable_determinism:
        parameters += ' --enable_determinism'
    if args.deterministic_seed is not None:
        parameters += f' --deterministic_seed {args.deterministic_seed}'

    context = BenchmarkRegistry.create_benchmark_context(args.model, parameters=parameters, framework=Framework.PYTORCH)
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    logger.info(f'Benchmark finished. Return code: {benchmark.return_code}')

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path('outputs') / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse benchmark results
    benchmark_results = json.loads(benchmark.serialized_result)
    benchmark_name = benchmark_results.get('name', f'pytorch-{args.model}')

    # Convert to results-summary.jsonl format (flattened keys)
    # Use format compatible with sb result commands: model-benchmarks:<category>/<benchmark>/<metric>
    summary = {}
    prefix = f'model-benchmarks:example:determinism/{benchmark_name}'
    if 'result' in benchmark_results:
        for metric, values in benchmark_results['result'].items():
            # Use first value if it's a list
            val = values[0] if isinstance(values, list) else values
            # Add _rank0 suffix to deterministic metrics for compatibility with rules
            if metric.startswith('deterministic_'):
                metric_key = f'{prefix}/{metric}_rank0'
            else:
                metric_key = f'{prefix}/{metric}'
            summary[metric_key] = val

    # Add node identifier
    summary['node'] = socket.gethostname()

    # Write results-summary.jsonl
    summary_file = output_dir / 'results-summary.jsonl'
    with open(summary_file, 'w') as f:
        f.write(json.dumps(summary))
    logger.info(f'Results saved to {summary_file}')

    # Also save full results for reference
    full_results_file = output_dir / 'results-full.json'
    with open(full_results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)

    if 'raw_data' in benchmark_results and 'deterministic_loss' in benchmark_results['raw_data']:
        num_checkpoints = len(benchmark_results['raw_data']['deterministic_loss'][0])
        logger.info(f'Periodic fingerprints collected at {num_checkpoints} checkpoints')

    logger.info(
        f'To generate baseline: sb result generate-baseline '
        f'--data-file {summary_file} --summary-rule-file summary-rules.yaml '
        f'--output-dir {output_dir}'
    )
    logger.info('To compare results between runs, use `sb result diagnosis` command.')


if __name__ == '__main__':
    main()
