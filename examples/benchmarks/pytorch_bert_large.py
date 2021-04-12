# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Model benchmark example for bert-large.

Commands to run:
  python3 examples/benchmarks/pytorch_bert_large.py (Single GPU)
  python3 -m torch.distributed.launch --nproc_per_node=8 examples/benchmarks/pytorch_bert_large.py (Distributed)
"""

from superbench.benchmarks import Framework, BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    # Create context for bert-large benchmark and run it for 120 * 2 seconds.
    context = BenchmarkRegistry.create_benchmark_context(
        'bert-large',
        parameters='--batch_size 1 --duration 120 --seq_len 8 --precision float32 --run_count 2',
        framework=Framework.PYTORCH
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
