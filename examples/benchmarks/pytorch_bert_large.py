# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for bert-large."""

from superbench.benchmarks import Platform, Framework, BenchmarkRegistry, BenchmarkContext
from superbench.common.utils import logger

if __name__ == '__main__':
    # Create context for bert-large benchmark and run it for 120 * 2 seconds.
    context = BenchmarkContext(
        'bert-large',
        Platform.CUDA,
        parameters='--batch_size=1 --duration=120 --seq_len=512 --precision=float32 --run_count=2',
        framework=Framework.PYTORCH
    )

    if BenchmarkRegistry.check_parameters(context):
        benchmark = BenchmarkRegistry.launch_benchmark(context)
        if benchmark:
            logger.info(
                'benchmark: {}, return code: {}, result: {}'.format(
                    benchmark.name, benchmark.return_code, benchmark.result
                )
            )
    else:
        logger.error('bert-large benchmark does not exist or context/parameters are invalid.')
