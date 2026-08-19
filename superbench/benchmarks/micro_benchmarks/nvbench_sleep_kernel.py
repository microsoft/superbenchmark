# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the NVBench Sleep Kernel benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks.nvbench_base import NvbenchBase


class NvbenchSleepKernel(NvbenchBase):
    """The NVBench Sleep Kernel benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'nvbench_sleep_kernel'

    def add_parser_arguments(self):
        """Add sleep-kernel specific arguments."""
        super().add_parser_arguments()

        # Sleep-kernel specific argument
        self._parser.add_argument(
            '--duration_us',
            type=str,
            default='[0,25,50,75,100]',
            help='Duration axis values in microseconds. Supports multiple formats: '
            '"50" (single value), "[25,50,75]" (list), "[25:75]" (range), "[0:50:10]" (range with step).',
        )

    def _extend_command(self, parts):
        """Add sleep-kernel specific arguments.

        Args:
            parts (list): Base command parts.

        Returns:
            list: Command parts including the duration axis.
        """
        parts.extend(['--axis', f'"Duration (us)={self._args.duration_us.strip()}"'])
        return parts

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

        self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        try:
            data = self._load_result_json(cmd_idx, raw_output)
            parsed_any = False
            for axes, summaries in self._iter_states(data):
                duration = axes.get('Duration (us)')
                if duration is None:
                    continue
                prefix = f'duration_us_{duration}'
                cpu_time = self._summary_value(summaries, 'nv/cold/time/cpu/mean') * 1e6
                gpu_time = self._summary_value(summaries, 'nv/cold/time/gpu/mean') * 1e6
                batch_gpu_time = self._summary_value(summaries, 'nv/batch/time/gpu/mean') * 1e6
                self._result.add_result(f'{prefix}_cpu_time', cpu_time)
                self._result.add_result(f'{prefix}_gpu_time', gpu_time)
                self._result.add_result(f'{prefix}_batch_gpu_time', batch_gpu_time)
                parsed_any = True

            if not parsed_any:
                raise ValueError('No valid result states parsed')

        except BaseException as e:
            self._handle_parsing_error(str(e), raw_output)
            return False

        return True


BenchmarkRegistry.register_benchmark('nvbench-sleep-kernel', NvbenchSleepKernel, platform=Platform.CUDA)
