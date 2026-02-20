# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the NVBench Auto Throughput benchmark."""

import re
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks.nvbench_base import NvbenchBase, parse_time_to_us


class NvbenchAutoThroughput(NvbenchBase):
    """The NVBench Auto Throughput benchmark class.

    This benchmark measures memory throughput and cache hit rates using CUPTI.
    It copies a 128 MiB buffer with configurable stride and items per thread.
    """

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._bin_name = 'nvbench_auto_throughput'

    def add_parser_arguments(self):
        """Add benchmark-specific arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--stride',
            type=str,
            default='[1:4]',
            help='Stride axis values. Supports: "2" (single), "[1,2,4]" (list), "[1:4]" (range), "[1:8:2]" (step).',
        )

        self._parser.add_argument(
            '--block_size',
            type=str,
            default='[128,256,512,1024]',
            help='Block size (threads per block). Supports: "256" (single), "[128,256,512,1024]" (list).',
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

        # Add stride axis argument
        parts.extend(['--axis', f'"Stride={self._args.stride.strip()}"'])

        # Add block size axis argument
        parts.extend(['--axis', f'"BlockSize={self._args.block_size.strip()}"'])

        # Finalize command
        self._commands = [' '.join(parts)]
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Parse raw results and save the summarized results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)

        try:
            # Pattern for throughput benchmark table output with CUPTI metrics
            # Table format:
            # | T | Stride | BlockSize | Elements | HBWPeak | LoadEff | StoreEff | L1HitRate | L2HitRate |
            # | Samples | Samples | CPU Time | Noise | GPU Time | Noise | Elem/s | Samples | Batch GPU |
            row_pat = (
                r'\|\s*(\d+)\s*\|'               # T (ItemsPerThread)
                r'\s*(\d+)\s*\|'                 # Stride
                r'\s*(\d+)\s*\|'                 # BlockSize
                r'\s*\d+\s*\|'                   # Elements (skip)
                r'\s*([\d.]+)%\s*\|'             # HBWPeak
                r'\s*([\d.]+)%\s*\|'             # LoadEff
                r'\s*([\d.]+)%\s*\|'             # StoreEff
                r'\s*([\d.]+)%\s*\|'             # L1HitRate
                r'\s*([\d.]+)%\s*\|'             # L2HitRate
                r'\s*\d+x\s*\|'                  # Samples CUPTI (skip)
                r'\s*\d+x\s*\|'                  # Samples Cold (skip)
                r'\s*([\d.]+\s*[μmun]?s)\s*\|'   # CPU Time
                r'\s*[\d.]+%\s*\|'               # CPU Noise (skip)
                r'\s*([\d.]+\s*[μmun]?s)\s*\|'   # GPU Time
                r'\s*[\d.]+%\s*\|'               # GPU Noise (skip)
                r'\s*([\d.]+)([TGMK]?)\s*\|'     # Elem/s (value and unit prefix)
                r'\s*\d+x\s*\|'                  # Samples Batch (skip)
                r'\s*([\d.]+\s*[μmun]?s)\s*\|'   # Batch GPU Time
            )

            parsed_any = False

            for line in raw_output.splitlines():
                line = line.strip()
                r = re.match(row_pat, line)
                if r:
                    (items_per_thread, stride, block_size,
                     hbw_peak, load_eff, store_eff, l1_hit, l2_hit,
                     cpu_time, gpu_time, elem_rate, elem_unit, batch_gpu) = r.groups()

                    prefix = f'ipt_{items_per_thread}_stride_{stride}_blk_{block_size}'

                    # Timing metrics (in microseconds)
                    self._result.add_result(f'{prefix}_cpu_time', parse_time_to_us(cpu_time))
                    self._result.add_result(f'{prefix}_gpu_time', parse_time_to_us(gpu_time))
                    self._result.add_result(f'{prefix}_batch_gpu_time', parse_time_to_us(batch_gpu))

                    # CUPTI metrics (percentages)
                    self._result.add_result(f'{prefix}_hbw_peak', float(hbw_peak))
                    self._result.add_result(f'{prefix}_load_eff', float(load_eff))
                    self._result.add_result(f'{prefix}_store_eff', float(store_eff))
                    self._result.add_result(f'{prefix}_l1_hit_rate', float(l1_hit))
                    self._result.add_result(f'{prefix}_l2_hit_rate', float(l2_hit))

                    # Memory throughput in GB/s
                    # Convert element rate to bandwidth: GB/s = (elements/s) * sizeof(int32) / 1e9
                    # The benchmark uses int32 (4 bytes per element)
                    elem_val = float(elem_rate)
                    unit_multipliers = {'T': 1e12, 'G': 1e9, 'M': 1e6, 'K': 1e3, '': 1.0}
                    elements_per_sec = elem_val * unit_multipliers.get(elem_unit, 1.0)
                    throughput_gbs = (elements_per_sec * 4) / 1e9  # 4 bytes per int32
                    self._result.add_result(f'{prefix}_throughput', throughput_gbs)

                    parsed_any = True

            if not parsed_any:
                raise ValueError('No valid result rows parsed')

        except BaseException as e:
            self._handle_parsing_error(str(e), raw_output)
            return False

        return True


BenchmarkRegistry.register_benchmark('nvbench-auto-throughput', NvbenchAutoThroughput, platform=Platform.CUDA)
