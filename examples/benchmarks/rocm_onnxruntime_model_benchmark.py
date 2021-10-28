# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Docker benchmark example for onnxruntime models.

Commands to run:
  python3 examples/benchmarks/rocm_onnxruntime_model_benchmark.py
"""

from superbench.benchmarks import BenchmarkRegistry, Framework, Platform
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context(
        'ort-models', platform=Platform.ROCM, framework=Framework.ONNXRUNTIME
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
