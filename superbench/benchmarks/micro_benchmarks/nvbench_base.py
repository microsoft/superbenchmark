# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for NVBench benchmarks."""

import json
import os
import re
import shlex
import shutil
import tempfile
from superbench.common.utils import logger
from superbench.benchmarks import ReturnCode
from superbench.benchmarks.micro_benchmarks.micro_base import MicroBenchmarkWithInvoke


def parse_time_to_us(raw: str) -> float:
    """Parse a time string like '123.45 us' or '1.5 s' to float microseconds."""
    raw = raw.strip()
    m = re.match(r'^([\d.]+)\s*([mun]?s)?$', raw)
    if not m:
        raise ValueError(f'Invalid time string: {raw!r}')
    val, unit = float(m.group(1)), (m.group(2) or 'us')
    if unit == 's':
        return val * 1e6
    elif unit == 'ns':
        return val / 1e3
    elif unit == 'ms':
        return val * 1e3
    return val


_NVBENCH_INT_VALUES_PATTERN = re.compile(
    r'(?:\d+|\[\s*\d+\s*(?:(?:,\s*\d+\s*)+|:\s*\d+\s*(?::\s*\d+\s*)?)?\])'
)


def parse_nvbench_int_values(value):
    """Validate an NVBench integer value specification."""
    # Accepted formats: '0', '[0,1,2]', '[0:4]', and '[0:4:2]' (range with step).
    if not _NVBENCH_INT_VALUES_PATTERN.fullmatch(value):
        raise ValueError(
            'Invalid NVBench integer values. Use a single value like "0", '
            'a list like "[0,1,2]", or a range like "[0:4]" or "[0:4:2]".'
        )
    return value


def _parse_devices(value):
    """Validate an NVBench device selection."""
    # Devices also accept 'all' and the legacy unbracketed list form '0,1,2'.
    if value == 'all' or re.fullmatch(r'\d+(?:,\d+)*', value):
        return value
    try:
        return parse_nvbench_int_values(value)
    except ValueError as error:
        raise ValueError(
            'Invalid --devices format. Use "all", GPU indices like "0,1,2", '
            'or an NVBench list/range like "[0,1,2]" or "[0:4]".'
        ) from error


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
        # Per-command NVBench --json output paths, populated during _preprocess.
        self._json_paths = []
        self._nvbench_tmp_dir = None

    def add_parser_arguments(self):
        """Add common NVBench arguments."""
        super().add_parser_arguments()

        # Device configuration
        self._parser.add_argument(
            '--devices',
            type=_parse_devices,
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
        self._parser.add_argument(
            '--output_dir',
            type=str,
            default=None,
            help='Directory for NVBench JSON result files. Defaults to a temporary directory that is '
            'removed after parsing. Set this to persist the raw NVBench JSON output.',
        )

    def _add_device_args(self, parts):
        """Add device configuration arguments to command parts."""
        if hasattr(self._args, 'devices') and self._args.devices is not None:
            if self._args.devices == 'all':
                parts.extend(['--devices', 'all'])
            else:
                parts.extend(['--devices', self._args.devices])

    def _add_benchmark_property_args(self, parts):
        """Add benchmark property arguments to command parts."""
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

    def _add_stopping_criteria_args(self, parts):
        """Add stopping criteria arguments to command parts."""
        if hasattr(self._args, 'timeout') and self._args.timeout is not None:
            parts.extend(['--timeout', str(self._args.timeout)])
        if hasattr(self._args, 'min_samples') and self._args.min_samples is not None:
            parts.extend(['--min-samples', str(self._args.min_samples)])
        if hasattr(self._args, 'stopping_criterion') and self._args.stopping_criterion:
            parts.extend(['--stopping-criterion', self._args.stopping_criterion])
            if self._args.stopping_criterion == 'stdrel':
                self._add_stdrel_args(parts)
            elif self._args.stopping_criterion == 'entropy':
                self._add_entropy_args(parts)

    def _add_stdrel_args(self, parts):
        """Add stdrel-specific stopping criterion arguments."""
        if hasattr(self._args, 'min_time') and self._args.min_time is not None:
            parts.extend(['--min-time', str(self._args.min_time)])
        if hasattr(self._args, 'max_noise') and self._args.max_noise is not None:
            parts.extend(['--max-noise', str(self._args.max_noise)])

    def _add_entropy_args(self, parts):
        """Add entropy-specific stopping criterion arguments."""
        if hasattr(self._args, 'max_angle') and self._args.max_angle is not None:
            parts.extend(['--max-angle', str(self._args.max_angle)])
        if hasattr(self._args, 'min_r2') and self._args.min_r2 is not None:
            parts.extend(['--min-r2', str(self._args.min_r2)])

    def _build_base_command(self):
        """Build the base nvbench command with common arguments.

        Returns:
            list: Command parts that can be extended by subclasses.
        """
        if not self._bin_name:
            raise ValueError('Subclass must set _bin_name')

        command = os.path.join(self._args.bin_dir, self._bin_name)
        parts = [command]

        self._add_device_args(parts)
        self._add_benchmark_property_args(parts)
        self._add_stopping_criteria_args(parts)

        return parts

    def _extend_command(self, parts):
        """Add benchmark-specific arguments to the command. Subclasses may override.

        Args:
            parts (list): Base command parts.

        Returns:
            list: Command parts including benchmark-specific arguments.
        """
        return parts

    def _preprocess(self):
        """Default preprocess implementation.

        Returns:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        parts = self._extend_command(self._build_base_command())
        self._commands = self._finalize_commands([' '.join(parts)])
        return True

    def _finalize_commands(self, commands):
        """Attach a unique NVBench --json output path to each command.

        Args:
            commands (list): Command strings without result-output arguments.

        Returns:
            list: Command strings with a `--json <path>` argument appended.
        """
        output_dir = getattr(self._args, 'output_dir', None)
        if output_dir:
            self._nvbench_tmp_dir = None
            base_dir = output_dir
        else:
            self._nvbench_tmp_dir = tempfile.mkdtemp(prefix='nvbench_')
            base_dir = self._nvbench_tmp_dir
        os.makedirs(base_dir, exist_ok=True)

        self._json_paths = []
        finalized = []
        for idx, command in enumerate(commands):
            json_path = os.path.join(base_dir, f'{self._bin_name}_{idx}.json')
            self._json_paths.append(json_path)
            finalized.append(f'{command} --json {shlex.quote(json_path)}')
        return finalized

    def _load_result_json(self, cmd_idx, raw_output):
        """Load NVBench JSON output for a command, preferring the written file.

        In production the NVBench binary writes the JSON file; when it is absent
        (e.g. unit tests) the provided raw_output is parsed instead.

        Args:
            cmd_idx (int): Command index.
            raw_output (str): Fallback JSON text.

        Returns:
            dict: Parsed NVBench JSON.
        """
        json_path = self._json_paths[cmd_idx] if cmd_idx < len(self._json_paths) else None
        raw_json = raw_output
        if json_path and os.path.isfile(json_path):
            with open(json_path, 'r') as file_handle:
                raw_json = file_handle.read()
            if not getattr(self._args, 'output_dir', None):
                try:
                    os.remove(json_path)
                except OSError:
                    pass
        if self._nvbench_tmp_dir and cmd_idx == len(self._json_paths) - 1:
            shutil.rmtree(self._nvbench_tmp_dir, ignore_errors=True)
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_json, self._args.log_raw_data)
        return json.loads(raw_json)

    @staticmethod
    def _iter_states(data):
        """Yield (axis_values, summaries_by_tag) for each non-skipped benchmark state.

        Args:
            data (dict): Parsed NVBench JSON.

        Yields:
            tuple: (dict of axis name -> value, dict of summary tag -> summary).
        """
        for benchmark in data.get('benchmarks', []):
            for state in benchmark.get('states', []):
                if state.get('is_skipped'):
                    continue
                summaries = {summary.get('tag'): summary for summary in state.get('summaries', [])}
                axes = {axis.get('name'): axis.get('value') for axis in (state.get('axis_values') or [])}
                yield axes, summaries

    @staticmethod
    def _summary_value(summaries, tag):
        """Extract the float 'value' entry of a summary by tag.

        Args:
            summaries (dict): Summary tag -> summary mapping.
            tag (str): Summary tag to look up.

        Returns:
            float: The summary value.
        """
        summary = summaries.get(tag)
        if summary is None:
            raise ValueError(f'Missing summary tag: {tag}')
        for entry in summary.get('data', []):
            if entry.get('name') == 'value':
                return float(entry['value'])
        raise ValueError(f'No value for summary tag: {tag}')

    def _handle_parsing_error(self, error_msg, raw_output):
        """Handle parsing errors consistently.

        Args:
            error_msg (str): Error message to log.
            raw_output (str): Raw output that failed to parse.
        """
        self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
        logger.error(
            'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                self._curr_run_index, self._name, raw_output, error_msg
            )
        )
