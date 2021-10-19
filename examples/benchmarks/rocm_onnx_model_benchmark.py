# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Docker benchmark example for onnx models.

Commands to run:
  python3 examples/benchmarks/rocm_onnx_model_benchmark.py
"""

from superbench.benchmarks import BenchmarkRegistry, Framework, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'model-benchmark', platform=Platform.ROCM, framework=Framework.ONNX
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
