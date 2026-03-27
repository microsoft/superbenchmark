# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DTK HPCG benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import GpuHpcgBenchmark


class DtkHpcgBenchmark(GpuHpcgBenchmark):
    """The DTK HPCG benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'run_rochpcg'


BenchmarkRegistry.register_benchmark('gpu-hpcg', DtkHpcgBenchmark, platform=Platform.DTK)
