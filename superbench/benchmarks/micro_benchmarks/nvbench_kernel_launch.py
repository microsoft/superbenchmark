import os
import re
from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke

def parse_time_to_us(raw: str) -> float:
    """Helper: parse '123.45 us', '678.9 ns', '0.12 ms' → float µs."""
    raw = raw.strip()
    if raw.endswith('%'):
        return float(raw[:-1])
    # split “value unit” or “valueunit”
    m = re.match(r'([\d.]+)\s*([mun]?s)?', raw)
    if not m:
        return float(raw)
    val, unit = float(m.group(1)), (m.group(2) or 'us')
    if unit == 'ns':  return val / 1e3
    if unit == 'ms':  return val * 1e3
    return val

class NvbenchKernelLaunch(MicroBenchmarkWithInvoke):
    """Nvbench benchmark wrapper for SuperBench."""
    def __init__(self, name, parameters=None):
        """Initialize the benchmark."""
        super().__init__(name, parameters)
        self._bin_name = "nvbench_kernel_launch"
    
    def add_parser_arguments(self):
        """
        Add NVBench CLI options (excluding Queries, Device modification, Output, Benchmark / Axis Specification):
          - Benchmark Properties (Omit --run-once, --disable-blocking-kernel, --profile)
          - Stopping Criteria
        """
        super().add_parser_arguments()

        # Benchmark Properties
        self._parser.add_argument(
            '--devices', '--device', '-d', type=str, default=None,
            help='Limit execution to one or more device IDs (comma-separated or "all").'
        )
        self._parser.add_argument(
            '--skip-time', type=float, default=-1.0,
            help='Skip a measurement when a warmup run executes in less than this (seconds).'
        )
        # With a threshold >0 and a recovery delay >0, NVBench will automatically pause 
        # and “wait for the card to warm up” back to a stable high‐clock state, giving you 
        # more consistent (and lower) timings that reflect the GPU’s true peak performance.
        self._parser.add_argument(
            '--throttle-threshold', type=float, default=75.0,
            help="GPU throttle threshold as percent of default clock rate. Disabled when nvbench::exec_tag::sync is used."
        )
        self._parser.add_argument(
            '--throttle-recovery-delay', type=float, default=0.05,
            help='Seconds to wait after throttle before resuming. '
            'Disabled when nvbench::exec_tag::sync is used.'
        )

        # Stopping Criteria
        self._parser.add_argument(
            '--timeout', type=int, default=15,
            help='Walltime timeout in seconds for each measurement.'
        )
        self._parser.add_argument(
            '--min-samples', type=int, default=10,
            help='Minimum number of samples per measurement before checking other criteria.'
        )
        self._parser.add_argument(
            '--stopping-criterion', type=str, default='stdrel',
            choices=['stdrel', 'entropy'],
            help='Stopping criterion to use after --min-samples is satisfied: '
            '"stdrel" or "entropy".'
        )
        # stdrel-specific
        self._parser.add_argument(
            '--min-time', type=float, default=0.5, 
            help='(stdrel) Minimum execution time accumulated per measurement (seconds).'
        )
        self._parser.add_argument(
            '--max-noise', type=float, default=0.5,
            help='(stdrel) Maximum relative standard deviation (%) before stopping.'
        )
        # entropy-specific
        self._parser.add_argument(
            '--max-angle', type=float, default=0.048,
            help='(entropy) Maximum linear regression angle of cumulative entropy.'
        )
        self._parser.add_argument(
            '--min-r2', type=float, default=0.36,
            help='(entropy) Minimum coefficient of determination (R²) for linear regression of cumulative entropy.'
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.
        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        bin_path = os.path.join(self._args.bin_dir, self._bin_name)
        parts = [bin_path]

        # Benchmark Properties (all optional)
        if self._args.devices:
            parts.extend(['--devices', self._args.devices])
        if self._args.skip_time >= 0:
            parts.extend(['--skip-time', str(self._args.skip_time)])
        if self._args.throttle_threshold > 0:
            parts.extend(['--throttle-threshold', str(self._args.throttle_threshold)])
        if self._args.throttle_recovery_delay > 0:
            parts.extend(['--throttle-recovery-delay', str(self._args.throttle_recovery_delay)])

        # Stopping Criteria (all optional)
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
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)
        try:
            gpu_section = r"### \[(\d+)\] NVIDIA"
            row_pat = (
                r"\| (\d+)x \| ([\d.]+ ?[mun]?s) \| ([\d.]+%) \| "
                r"([\d.]+ ?[mun]?s) \| ([\d.]+%) \| (\d+)x \| *([\d.]+ ?[mun]?s) \|"
            )
            current = None
            parsed_any = False  # Track if any valid rows are parsed
            for line in raw_output.splitlines():
                line = line.strip()
                g = re.match(gpu_section, line)
                if g:
                    current = f"gpu_{g.group(1)}"
                    continue
                r = re.match(row_pat, line)
                if r and current:
                    self._result.add_result(f"{current}_samples", int(r.group(1)))
                    self._result.add_result(f"{current}_cpu_time", parse_time_to_us(r.group(2)))
                    self._result.add_result(f"{current}_cpu_noise", float(r.group(3)[:-1]))
                    self._result.add_result(f"{current}_gpu_time", parse_time_to_us(r.group(4)))
                    self._result.add_result(f"{current}_gpu_noise", float(r.group(5)[:-1]))
                    self._result.add_result(f"{current}_batch_samples", int(r.group(6)))
                    self._result.add_result(f"{current}_batch_gpu_time", parse_time_to_us(r.group(7)))
                    parsed_any = True
            if not parsed_any:
                logger.error("No valid rows parsed from the raw output.")
                raise RuntimeError("No valid rows parsed")
        except Exception as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                f"Invalid result format - round:{self._curr_run_index}, bench:{self._name}, msg:{e}\n{raw_output}"
            )
            return False
        return True

# Register the benchmark
BenchmarkRegistry.register_benchmark("nvbench-kernel-launch", NvbenchKernelLaunch, platform=Platform.CUDA)