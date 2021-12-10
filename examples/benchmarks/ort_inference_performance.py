# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for ONNXRuntime inference performance.

Commands to run:
    python3 examples/benchmarks/ort_inference_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
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
