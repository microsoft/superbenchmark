# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the NVBench Sleep Kernel benchmark."""

import re
import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks.micro_base import MicroBenchmarkWithInvoke


class NvbenchSleepKernel(MicroBenchmarkWithInvoke):
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
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--devices',
            type=str,
            default=None,
            help='Device list to run the benchmark, e.g., "0,1,2,3" or "all".',
        )
        self._parser.add_argument(
            '--duration_us',
            type=str,
            default='[0,25,50,75,100]',
            help='Duration axis values in microseconds, e.g., "[0,25,50,75,100]".',
        )
        self._parser.add_argument(
            '--skip_time',
            type=float,
            default=-1.0,
            help='Skip time in seconds.',
        )
        self._parser.add_argument(
            '--throttle_threshold',
            type=float,
            default=75.0,
            help='Throttle threshold percentage.',
        )
        self._parser.add_argument(
            '--throttle_recovery_delay',
            type=float,
            default=0.05,
            help='Throttle recovery delay in seconds.',
        )
        self._parser.add_argument(
            '--run_once',
            action='store_true',
            help='Run once flag.',
        )
        self._parser.add_argument(
            '--disable_blocking_kernel',
            action='store_true',
            help='Disable blocking kernel flag.',
        )
        self._parser.add_argument(
            '--profile',
            action='store_true',
            help='Enable profiling flag.',
        )
        self._parser.add_argument(
            '--timeout',
            type=int,
            default=15,
            help='Timeout in seconds.',
        )
        self._parser.add_argument(
            '--min_samples',
            type=int,
            default=None,
            help='Minimum number of samples.',
        )
        self._parser.add_argument(
            '--stopping_criterion',
            type=str,
            default='stdrel',
            choices=['stdrel', 'entropy'],
            help='Stopping criterion.',
        )
        self._parser.add_argument(
            '--min_time',
            type=float,
            default=None,
            help='Minimum time for stdrel stopping criterion.',
        )
        self._parser.add_argument(
            '--max_noise',
            type=float,
            default=None,
            help='Maximum noise for stdrel stopping criterion.',
        )
        self._parser.add_argument(
            '--max_angle',
            type=float,
            default=None,
            help='Maximum angle for entropy stopping criterion.',
        )
        self._parser.add_argument(
            '--min_r2',
            type=float,
            default=None,
            help='Minimum R-squared for entropy stopping criterion.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        command = os.path.join(self._args.bin_dir, self._bin_name)
        parts = [command]

        # Basic configuration
        if self._args.devices is not None:
            if self._args.devices == 'all':
                parts.extend(['--devices', 'all'])
            else:
                parts.extend(['--devices', self._args.devices])

        # Duration axis
        parts.extend(['--axis', f'"Duration (us)={self._args.duration_us}"'])

        # Performance configuration
        if self._args.skip_time >= 0:
            parts.extend(['--skip-time', str(self._args.skip_time)])
        parts.extend(['--throttle-threshold', str(self._args.throttle_threshold)])
        parts.extend(['--throttle-recovery-delay', str(self._args.throttle_recovery_delay)])
        if self._args.run_once:
            parts.append('--run-once')
        if self._args.disable_blocking_kernel:
            parts.append('--disable-blocking-kernel')
        if self._args.profile:
            parts.append('--profile')

        # Stopping criteria
        if self._args.timeout is not None:
            parts.extend(['--timeout', str(self._args.timeout)])
        if self._args.min_samples is not None:
            parts.extend(['--min-samples', str(self._args.min_samples)])
        if self._args.stopping_criterion:
            parts.extend(['--stopping-criterion', self._args.stopping_criterion])
            if self._args.stopping_criterion == 'stdrel':
                if self._args.min_time is not None:
                    parts.extend(['--min-time', str(self._args.min_time)])
                if self._args.max_noise is not None:
                    parts.extend(['--max-noise', str(self._args.max_noise)])
            elif self._args.stopping_criterion == 'entropy':
                if self._args.max_angle is not None:
                    parts.extend(['--max-angle', str(self._args.max_angle)])
                if self._args.min_r2 is not None:
                    parts.extend(['--min-r2', str(self._args.min_r2)])

        # finalize command
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
        logger.debug(f"Processing raw result for command index {cmd_idx}.")
        logger.debug(f"Raw output:\n{raw_output}")

        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)
        try:
            gpu_section = r"### \[(\d+)\] NVIDIA"
            row_pat = (
                r"\|\s*(\d+)\s*\|\s*(\d+)x\s*\|\s*([\d.]+ ?[mun]?s)\s*\|\s*([\d.]+%)\s*\|\s*"
                r"([\d.]+ ?[mun]?s)\s*\|\s*([\d.]+%)\s*\|\s*(\d+)x\s*\|\s*([\d.]+ ?[mun]?s)\s*\|"
            )
            current = None
            parsed_any = False
            for line in raw_output.splitlines():
                line = line.strip()
                logger.debug(f"Processing line: {line}")
                g = re.match(gpu_section, line)
                if g:
                    current = f"gpu_{g.group(1)}"
                    logger.debug(f"Found GPU section: {current}")
                    continue
                r = re.match(row_pat, line)
                if r and current:
                    logger.debug(f"Matched row: {r.groups()}")
                    duration_us, samples, cpu_time, cpu_noise, gpu_time, gpu_noise, batch_samples, batch_gpu = r.groups()
                    self._result.add_result(f'{current}_duration_us_{duration_us}_samples', int(samples))
                    self._result.add_result(f'{current}_duration_us_{duration_us}_cpu_time', self._parse_time_value(cpu_time))
                    self._result.add_result(f'{current}_duration_us_{duration_us}_cpu_noise', self._parse_percentage(cpu_noise))
                    self._result.add_result(f'{current}_duration_us_{duration_us}_gpu_time', self._parse_time_value(gpu_time))
                    self._result.add_result(f'{current}_duration_us_{duration_us}_gpu_noise', self._parse_percentage(gpu_noise))
                    self._result.add_result(f'{current}_duration_us_{duration_us}_batch_samples', int(batch_samples))
                    self._result.add_result(f'{current}_duration_us_{duration_us}_batch_gpu', self._parse_time_value(batch_gpu))
                    parsed_any = True
            if not parsed_any:
                raise RuntimeError("No valid rows parsed")
        except Exception as e:
            logger.error(f"Error processing raw result: {e}")
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            return False
        return True

    def _parse_time_value(self, time_str):
        """Parse time string to microseconds.

        Args:
            time_str (str): Time string like "25.321 us", "1.234 ms", etc.

        Returns:
            float: Time value in microseconds.
        """
        time_str = time_str.strip()
        if time_str.endswith('us'):
            return float(time_str[:-2].strip())
        elif time_str.endswith('ms'):
            return float(time_str[:-2].strip()) * 1000
        elif time_str.endswith('ns'):
            return float(time_str[:-2].strip()) / 1000
        elif time_str.endswith('s'):
            return float(time_str[:-1].strip()) * 1000000
        else:
            # Assume microseconds if no unit
            return float(time_str)

    def _parse_percentage(self, percent_str):
        """Parse percentage string to float.

        Args:
            percent_str (str): Percentage string like "0.93%".

        Returns:
            float: Percentage value as float.
        """
        return float(percent_str[:-1].strip())


BenchmarkRegistry.register_benchmark('nvbench-sleep-kernel', NvbenchSleepKernel, platform=Platform.CUDA)
