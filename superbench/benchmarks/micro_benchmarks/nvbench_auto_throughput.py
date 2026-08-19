# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module of the NVBench Auto Throughput benchmark."""

from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks.nvbench_base import NvbenchBase


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

    def _extend_command(self, parts):
        """Add auto-throughput specific axis arguments.

        Args:
            parts (list): Base command parts.

        Returns:
            list: Command parts including the stride and block-size axes.
        """
        parts.extend(['--axis', f'"Stride={self._args.stride.strip()}"'])
        parts.extend(['--axis', f'"BlockSize={self._args.block_size.strip()}"'])
        return parts

    def _process_raw_result(self, cmd_idx, raw_output):
        """Parse raw results and save the summarized results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        # CUPTI metrics are stored as fractions [0, 1] in the JSON and reported as percentages.
        cupti_tags = {
            'hbw_peak': 'nv/cupti/dram__throughput.avg.pct_of_peak_sustained_elapsed',
            'load_eff': 'nv/cupti/smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct',
            'store_eff': 'nv/cupti/smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct',
            'l1_hit_rate': 'nv/cupti/l1tex__t_sector_hit_rate.pct',
            'l2_hit_rate': 'nv/cupti/lts__t_sector_hit_rate.pct',
        }
        try:
            data = self._load_result_json(cmd_idx, raw_output)
            parsed_any = False
            for axes, summaries in self._iter_states(data):
                items_per_thread = axes.get('T')
                stride = axes.get('Stride')
                block_size = axes.get('BlockSize')
                if items_per_thread is None or stride is None or block_size is None:
                    continue
                prefix = f'ipt_{items_per_thread}_stride_{stride}_blk_{block_size}'

                # Timing metrics (microseconds)
                cpu_time = self._summary_value(summaries, 'nv/cold/time/cpu/mean') * 1e6
                gpu_time = self._summary_value(summaries, 'nv/cold/time/gpu/mean') * 1e6
                batch_gpu_time = self._summary_value(summaries, 'nv/batch/time/gpu/mean') * 1e6
                self._result.add_result(f'{prefix}_cpu_time', cpu_time)
                self._result.add_result(f'{prefix}_gpu_time', gpu_time)
                self._result.add_result(f'{prefix}_batch_gpu_time', batch_gpu_time)

                # CUPTI metrics (fraction -> percentage)
                for metric, tag in cupti_tags.items():
                    self._result.add_result(f'{prefix}_{metric}', self._summary_value(summaries, tag) * 100)

                # Memory throughput in GB/s: (elements/s) * sizeof(int32) / 1e9, int32 is 4 bytes.
                item_rate = self._summary_value(summaries, 'nv/cold/bw/item_rate')
                self._result.add_result(f'{prefix}_throughput', (item_rate * 4) / 1e9)

                parsed_any = True

            if not parsed_any:
                raise ValueError('No valid result states parsed')

        except BaseException as e:
            self._handle_parsing_error(str(e), raw_output)
            return False

        return True


BenchmarkRegistry.register_benchmark('nvbench-auto-throughput', NvbenchAutoThroughput, platform=Platform.CUDA)
