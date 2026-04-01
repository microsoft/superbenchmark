# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for ONNXRuntime inference performance.

Commands to run:
  In-house models:
    python3 examples/benchmarks/ort_inference_performance.py
    python3 examples/benchmarks/ort_inference_performance.py --model_source in-house

  HuggingFace models:
    python3 examples/benchmarks/ort_inference_performance.py \
      --model_source huggingface --model_identifier bert-base-uncased
    python3 examples/benchmarks/ort_inference_performance.py \
      --model_source huggingface --model_identifier microsoft/resnet-50

Environment variables:
  HF_TOKEN: HuggingFace token for gated models (optional)
"""

import argparse

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger


def run_inhouse_benchmark():
    """Run ORT inference with in-house torchvision models."""
    context = BenchmarkRegistry.create_benchmark_context(
        'ort-inference', platform=Platform.CUDA, parameters='--pytorch_models resnet50 resnet101 --precision float16'
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
    return benchmark


def run_huggingface_benchmark(model_identifier, precision='float16', batch_size=32, seq_length=512):
    """Run ORT inference with a HuggingFace model.

    Args:
        model_identifier: HuggingFace model ID (e.g., 'bert-base-uncased').
        precision: Inference precision ('float32', 'float16', 'int8').
        batch_size: Batch size for inference.
        seq_length: Sequence length for transformer models.
    """
    parameters = (
        f'--model_source huggingface '
        f'--model_identifier {model_identifier} '
        f'--precision {precision} '
        f'--batch_size {batch_size} '
        f'--seq_length {seq_length}'
    )

    logger.info(f'Running ORT inference benchmark with HuggingFace model: {model_identifier}')

    context = BenchmarkRegistry.create_benchmark_context('ort-inference', platform=Platform.CUDA, parameters=parameters)
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
    return benchmark


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORT inference benchmark')
    parser.add_argument(
        '--model_source',
        type=str,
        default='in-house',
        choices=['in-house', 'huggingface'],
        help='Source of the model: in-house (default) or huggingface'
    )
    parser.add_argument(
        '--model_identifier', type=str, default='bert-base-uncased', help='HuggingFace model identifier'
    )
    parser.add_argument('--precision', type=str, default='float16', choices=['float32', 'float16', 'int8'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_length', type=int, default=512)
    args = parser.parse_args()

    if args.model_source == 'huggingface':
        run_huggingface_benchmark(args.model_identifier, args.precision, args.batch_size, args.seq_length)
    else:
        run_inhouse_benchmark()
