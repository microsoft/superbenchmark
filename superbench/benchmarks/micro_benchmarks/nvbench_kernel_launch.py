# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the NVBench Kernel Launch benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks.nvbench_base import NvbenchBase


class NvbenchKernelLaunch(NvbenchBase):
    """The NVBench Kernel Launch benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._bin_name = 'nvbench_kernel_launch'

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        try:
            data = self._load_result_json(cmd_idx, raw_output)
            parsed_any = False
            for _axes, summaries in self._iter_states(data):
                cpu_time = self._summary_value(summaries, 'nv/cold/time/cpu/mean') * 1e6
                gpu_time = self._summary_value(summaries, 'nv/cold/time/gpu/mean') * 1e6
                batch_gpu_time = self._summary_value(summaries, 'nv/batch/time/gpu/mean') * 1e6
                self._result.add_result('cpu_time', cpu_time)
                self._result.add_result('gpu_time', gpu_time)
                self._result.add_result('batch_gpu_time', batch_gpu_time)
                parsed_any = True

            if not parsed_any:
                raise ValueError('No valid result states parsed')

        except BaseException as e:
            self._handle_parsing_error(str(e), raw_output)
            return False

        return True


BenchmarkRegistry.register_benchmark('nvbench-kernel-launch', NvbenchKernelLaunch, platform=Platform.CUDA)
