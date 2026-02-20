# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the NVBench Sleep Kernel benchmark."""

import re
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks.nvbench_base import NvbenchBase, parse_time_to_us


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

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Build base command with common nvbench arguments
        parts = self._build_base_command()

        # Add sleep-kernel specific arguments
        parts.extend(['--axis', f'"Duration (us)={self._args.duration_us.strip()}"'])

        # Finalize command
        self._commands = [' '.join(parts)]
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

        self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)
        try:
            gpu_section = r'### \[(\d+)\] NVIDIA'
            # Regex pattern to handle different time units and flexible spacing
            row_pat = (
                r'\|\s*([0-9]+)\s*\|\s*'    # Duration (us)
                r'([0-9]+)x\s*\|\s*'    # Samples
                r'([\d.]+\s*[μmun]?s)\s*\|\s*'    # CPU Time (μs, ns, ms, us, s)
                r'([\d.]+%)\s*\|\s*'    # CPU Noise percentage
                r'([\d.]+\s*[μmun]?s)\s*\|\s*'    # GPU Time
                r'([\d.]+%)\s*\|\s*'    # GPU Noise percentage
                r'([0-9]+)x\s*\|\s*'    # Batch Samples
                r'([\d.]+\s*[μmun]?s)\s*\|'    # Batch GPU Time
            )
            parsed_any = False
            for line in raw_output.splitlines():
                line = line.strip()
                g = re.match(gpu_section, line)
                if g:
                    continue
                r = re.match(row_pat, line)
                if r:
                    duration_us, samples, cpu_time, cpu_noise, gpu_time, gpu_noise, batch_samples, batch_gpu = r.groups(
                    )
                    self._result.add_result(f'duration_us_{duration_us}_cpu_time', parse_time_to_us(cpu_time))
                    self._result.add_result(f'duration_us_{duration_us}_gpu_time', parse_time_to_us(gpu_time))
                    self._result.add_result(
                        f'duration_us_{duration_us}_batch_gpu_time', parse_time_to_us(batch_gpu)
                    )
                    parsed_any = True

            if not parsed_any:
                raise ValueError('No valid result rows parsed')

        except BaseException as e:
            self._handle_parsing_error(str(e), raw_output)
            return False

        return True


BenchmarkRegistry.register_benchmark('nvbench-sleep-kernel', NvbenchSleepKernel, platform=Platform.CUDA)
