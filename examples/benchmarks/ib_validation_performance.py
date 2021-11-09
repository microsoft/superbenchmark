# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Micro benchmark example for IB validation performance between nodes.

Commands to run:
  mpirun -np 2 -H node0:1,node1:1  -mca pml ob1 --mca btl ^openib \
      -mca btl_tcp_if_exclude lo,docker0 -mca coll_hcoll_enable 0 \
          -x LD_LIBRARY_PATH -x PATH python examples/benchmarks/ib_traffic_performance.py
"""

from superbench.benchmarks import BenchmarkRegistry
from superbench.common.utils import logger

if __name__ == '__main__':
    context = BenchmarkRegistry.create_benchmark_context('ib-traffic')

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            'benchmark: {}, return code: {}, result: {}'.format(
                benchmark.name, benchmark.return_code, benchmark.result
            )
        )
