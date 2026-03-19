# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU Stream Performance benchmark."""

import csv
import io
import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuStreamBenchmark(MicroBenchmarkWithInvoke):
    """The GPU stream performance benchmark class."""
    _function_metric_map = {
        'Copy': 'COPY',
        'Mul': 'MUL',
        'Add': 'ADD',
        'Triad': 'TRIAD',
        'Dot': 'DOT',
    }
    _phase_metric_map = {
        'Init': 'INIT',
        'Read': 'READ',
    }

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'hip-stream'
        self.__bin_path = None

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--array_size',
            type=int,
            default=268435456,
            required=False,
            help='Number of elements in array.',
        )

        self._parser.add_argument(
            '--num_loops',
            type=int,
            default=100,
            required=False,
            help='Number of benchmark runs, mapping to --numtimes in BabelStream.',
        )

        self._parser.add_argument(
            '--precision',
            type=str,
            default='double',
            choices=['double', 'float'],
            required=False,
            help='Data type for benchmark.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        args = f'--arraysize {self._args.array_size} --numtimes {self._args.num_loops} --csv'
        if self._args.precision == 'float':
            args += ' --float'
        self._commands = ['{} {}'.format(self.__bin_path, args)]

        return True

    def _get_device_name(self, raw_output):
        """Extract device name from BabelStream output when available."""
        for line in raw_output.splitlines():
            if line.startswith('Using HIP device '):
                return line[len('Using HIP device '):].strip()
        return 'Unknown'

    @staticmethod
    def _mbps_to_gbps(value):
        """Convert MB/s to GB/s."""
        return float(value) / 1000

    def _parse_csv_phase_rows(self, raw_output):
        """Extract phase rows from BabelStream CSV output."""
        lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
        header = 'phase,n_elements,sizeof,max_mbytes_per_sec,runtime'
        if header not in lines:
            raise ValueError('No phase CSV header found in output.')

        start_idx = lines.index(header)
        csv_content = '\n'.join(lines[start_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))

        phase_rows = []
        for row in reader:
            phase_name = row.get('phase', '').strip()
            if phase_name in self._phase_metric_map:
                metric_tag = self._phase_metric_map[phase_name]
                array_size = int(row['n_elements'])
                phase_rows.append({
                    'metric_name': self._get_phase_bw_metric_name(metric_tag, array_size),
                    'value': self._mbps_to_gbps(row['max_mbytes_per_sec']),
                })
                phase_rows.append({
                    'metric_name': self._get_phase_time_metric_name(metric_tag, array_size),
                    'value': float(row['runtime']),
                })

        if not phase_rows:
            raise ValueError('No valid phase rows found in CSV output.')

        return phase_rows

    def _parse_csv_function_rows(self, raw_output):
        """Extract function rows from BabelStream CSV output."""
        lines = [line.strip() for line in raw_output.strip().splitlines() if line.strip()]
        header = 'function,num_times,n_elements,sizeof,max_mbytes_per_sec,min_runtime,max_runtime,avg_runtime'
        if header not in lines:
            raise ValueError('No function CSV header found in output.')

        start_idx = lines.index(header)
        csv_content = '\n'.join(lines[start_idx:])
        reader = csv.DictReader(io.StringIO(csv_content))

        function_rows = []
        for row in reader:
            function_name = row.get('function', '').strip()
            if function_name in self._function_metric_map:
                metric_tag = self._function_metric_map[function_name]
                array_size = int(row['n_elements'])
                function_rows.append({
                    'metric_name': self._get_function_bw_metric_name(metric_tag, array_size),
                    'value': self._mbps_to_gbps(row['max_mbytes_per_sec']),
                })
                function_rows.append({
                    'metric_name': self._get_function_time_metric_name(metric_tag, array_size, 'min'),
                    'value': float(row['min_runtime']),
                })
                function_rows.append({
                    'metric_name': self._get_function_time_metric_name(metric_tag, array_size, 'max'),
                    'value': float(row['max_runtime']),
                })
                function_rows.append({
                    'metric_name': self._get_function_time_metric_name(metric_tag, array_size, 'avg'),
                    'value': float(row['avg_runtime']),
                })

        if not function_rows:
            raise ValueError('No valid function rows found in CSV output.')

        return function_rows

    def _get_phase_bw_metric_name(self, metric_tag, array_size):
        """Build phase bandwidth metric name."""
        return 'STREAM_{}_{}_array_{}_bw'.format(metric_tag, self._args.precision, array_size)

    def _get_phase_time_metric_name(self, metric_tag, array_size):
        """Build phase runtime metric name."""
        return 'STREAM_{}_{}_array_{}_time'.format(metric_tag, self._args.precision, array_size)

    def _get_function_bw_metric_name(self, metric_tag, array_size):
        """Build function bandwidth metric name."""
        return 'STREAM_{}_{}_array_{}_bw'.format(metric_tag, self._args.precision, array_size)

    def _get_function_time_metric_name(self, metric_tag, array_size, metric_type):
        """Build function runtime metric name."""
        return 'STREAM_{}_{}_array_{}_time_{}'.format(metric_tag, self._args.precision, array_size, metric_type)

    def _format_device_output(self, device_name, metrics):
        """Render one device section in a human-readable format."""
        metric_width = max(len(metric['metric_name']) for metric in metrics)
        output_lines = ['Device: {}'.format(device_name)]
        for metric in metrics:
            output_lines.append('{:<{width}}  {:.6f}'.format(metric['metric_name'], metric['value'], width=metric_width))
        return output_lines

    def _get_text_output_header(self):
        """Render benchmark metadata in the text output header."""
        return [
            'STREAM Benchmark (BabelStream backend)',
            'Array size(elements): {}'.format(self._args.array_size),
            'Number of loops: {}'.format(self._args.num_loops),
            'Precision: {}'.format(self._args.precision),
            'Bandwidth unit: GB/s (converted from MB/s / 1000)',
        ]

    def _parse_device_output(self, raw_output):
        """Parse one device output and return rendered lines and parsed metrics."""
        device_name = self._get_device_name(raw_output)
        metrics = self._parse_csv_phase_rows(raw_output) + self._parse_csv_function_rows(raw_output)
        rendered_lines = self._get_text_output_header()
        rendered_lines.append('')
        rendered_lines.extend(self._format_device_output(device_name, metrics))
        return rendered_lines, metrics

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
            rendered_lines, metrics = self._parse_device_output(raw_output)
            self._result.add_raw_data('raw_output_' + str(cmd_idx), '\n'.join(rendered_lines), self._args.log_raw_data)
            for metric in metrics:
                self._result.add_result(metric['metric_name'], metric['value'])
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('gpu-stream', GpuStreamBenchmark)
