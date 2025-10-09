# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for NVBench benchmarks."""

import os
import re
from superbench.common.utils import logger
from superbench.benchmarks import ReturnCode
from superbench.benchmarks.micro_benchmarks.micro_base import MicroBenchmarkWithInvoke


def parse_time_to_us(raw: str) -> float:
    """Helper: parse '123.45 us', '678.9 ns', '0.12 ms' → float µs."""
    raw = raw.strip()
    if raw.endswith('%'):
        return float(raw[:-1])
    # split "value unit" or "valueunit"
    m = re.match(r'([\d.]+)\s*([mun]?s)?', raw)
    if not m:
        return float(raw)
    val, unit = float(m.group(1)), (m.group(2) or 'us')
    if unit == 'ns':
        return val / 1e3
    if unit == 'ms':
        return val * 1e3
    return val


class NvbenchBase(MicroBenchmarkWithInvoke):
    """Base class for NVBench benchmarks with common functionality."""

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        # Subclasses should set this
        self._bin_name = None

    def add_parser_arguments(self):
        """Add common NVBench arguments."""
        super().add_parser_arguments()

        # Device configuration
        self._parser.add_argument(
            '--devices',
            type=str,
            default=None,
            help='Device list to run the benchmark, e.g., "0,1,2,3" or "all".',
        )

        # Benchmark Properties
        self._parser.add_argument(
            '--skip-time',
            type=float,
            default=-1.0,
            help='Skip time in seconds.',
        )
        self._parser.add_argument(
            '--throttle-threshold',
            type=float,
            default=75.0,
            help='Throttle threshold percentage.',
        )
        self._parser.add_argument(
            '--throttle-recovery-delay',
            type=float,
            default=0.05,
            help='Throttle recovery delay in seconds.',
        )
        self._parser.add_argument(
            '--run-once',
            action='store_true',
            help='Run once flag.',
        )
        self._parser.add_argument(
            '--disable-blocking-kernel',
            action='store_true',
            help='Disable blocking kernel flag.',
        )
        self._parser.add_argument(
            '--profile',
            action='store_true',
            help='Enable profiling flag.',
        )

        # Stopping Criteria
        self._parser.add_argument(
            '--timeout',
            type=int,
            default=15,
            help='Timeout in seconds.',
        )
        self._parser.add_argument(
            '--min-samples',
            type=int,
            default=10,
            help='Minimum number of samples.',
        )
        self._parser.add_argument(
            '--stopping-criterion',
            type=str,
            default='stdrel',
            choices=['stdrel', 'entropy'],
            help='Stopping criterion.',
        )
        # stdrel-specific
        self._parser.add_argument(
            '--min-time',
            type=float,
            default=0.5,
            help='Minimum time for stdrel stopping criterion.',
        )
        self._parser.add_argument(
            '--max-noise',
            type=float,
            default=0.5,
            help='Maximum noise for stdrel stopping criterion.',
        )
        # entropy-specific
        self._parser.add_argument(
            '--max-angle',
            type=float,
            default=0.048,
            help='Maximum angle for entropy stopping criterion.',
        )
        self._parser.add_argument(
            '--min-r2',
            type=float,
            default=0.36,
            help='Minimum R-squared for entropy stopping criterion.',
        )

    def _build_base_command(self):
        """Build the base nvbench command with common arguments.
        
        Returns:
            list: Command parts that can be extended by subclasses.
        """
        if not self._bin_name:
            raise ValueError("Subclass must set _bin_name")
            
        command = os.path.join(self._args.bin_dir, self._bin_name)
        parts = [command]

        # Device configuration - in distributed mode, let SuperBench handle device assignment
        # Only add --devices if explicitly specified
        if hasattr(self._args, 'devices') and self._args.devices is not None:
            if self._args.devices == 'all':
                parts.extend(['--devices', 'all'])
            else:
                parts.extend(['--devices', self._args.devices])

        # Benchmark Properties
        if hasattr(self._args, 'skip_time') and self._args.skip_time >= 0:
            parts.extend(['--skip-time', str(self._args.skip_time)])
        if hasattr(self._args, 'throttle_threshold') and self._args.throttle_threshold > 0:
            parts.extend(['--throttle-threshold', str(self._args.throttle_threshold)])
        if hasattr(self._args, 'throttle_recovery_delay') and self._args.throttle_recovery_delay > 0:
            parts.extend(['--throttle-recovery-delay', str(self._args.throttle_recovery_delay)])
        if hasattr(self._args, 'run_once') and self._args.run_once:
            parts.append('--run-once')
        if hasattr(self._args, 'disable_blocking_kernel') and self._args.disable_blocking_kernel:
            parts.append('--disable-blocking-kernel')
        if hasattr(self._args, 'profile') and self._args.profile:
            parts.append('--profile')

        # Stopping criteria
        if hasattr(self._args, 'timeout') and self._args.timeout is not None:
            parts.extend(['--timeout', str(self._args.timeout)])
        if hasattr(self._args, 'min_samples') and self._args.min_samples is not None:
            parts.extend(['--min-samples', str(self._args.min_samples)])
        if hasattr(self._args, 'stopping_criterion') and self._args.stopping_criterion:
            parts.extend(['--stopping-criterion', self._args.stopping_criterion])
            if self._args.stopping_criterion == 'stdrel':
                if hasattr(self._args, 'min_time') and self._args.min_time is not None:
                    parts.extend(['--min-time', str(self._args.min_time)])
                if hasattr(self._args, 'max_noise') and self._args.max_noise is not None:
                    parts.extend(['--max-noise', str(self._args.max_noise)])
            elif self._args.stopping_criterion == 'entropy':
                if hasattr(self._args, 'max_angle') and self._args.max_angle is not None:
                    parts.extend(['--max-angle', str(self._args.max_angle)])
                if hasattr(self._args, 'min_r2') and self._args.min_r2 is not None:
                    parts.extend(['--min-r2', str(self._args.min_r2)])

        return parts

    def _preprocess(self):
        """Default preprocess implementation. Can be overridden by subclasses.
        
        Returns:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Build base command - subclasses can override this method to add specific arguments
        parts = self._build_base_command()
        
        # Finalize command
        self._commands = [' '.join(parts)]
        return True

    def _parse_time_value(self, time_str):
        """Parse time string to microseconds.
        
        Args:
            time_str (str): Time string like "123.45 us", "678.9 ns", etc.
            
        Returns:
            float: Time in microseconds.
        """
        return parse_time_to_us(time_str)

    def _parse_percentage(self, percent_str):
        """Parse percentage string to float.
        
        Args:
            percent_str (str): Percentage string like "12.34%"
            
        Returns:
            float: Percentage value as float.
        """
        if isinstance(percent_str, str) and percent_str.endswith('%'):
            return float(percent_str[:-1])
        return float(percent_str)

    def _handle_parsing_error(self, error_msg, raw_output):
        """Handle parsing errors consistently.
        
        Args:
            error_msg (str): Error message to log.
            raw_output (str): Raw output that failed to parse.
        """
        self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
        logger.error(
            f"Invalid result format - round:{self._curr_run_index}, bench:{self._name}, msg:{error_msg}\n{raw_output}"
        )