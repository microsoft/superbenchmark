# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the NV Bandwidth Test."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class NvBandwidthBenchmark(MicroBenchmarkWithInvoke):
    """The NV Bandwidth Test benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'nvbandwidth'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--buffer_size',
            type=int,
            default=64,
            required=False,
            help='Memcpy buffer size in MiB. Default is 64.',
        )

        self._parser.add_argument(
            '--test_cases',
            type=str,
            default='',
            required=False,
            help=(
                'Specify the test case(s) to run, either by name or index. By default, all test cases are executed. '
                'Example: --test_cases 0,1,2,19,20'
            ),
        )

        self._parser.add_argument(
            '--skip_verification',
            action='store_true',
            help='Skips data verification after copy. Default is False.',
        )

        self._parser.add_argument(
            '--disable_affinity',
            action='store_true',
            help='Disable automatic CPU affinity control. Default is False.',
        )

        self._parser.add_argument(
            '--use_mean',
            action='store_true',
            help='Use mean instead of median for results. Default is False.',
        )

        self._parser.add_argument(
            '--num_loops',
            type=int,
            default=3,
            required=False,
            help='Iterations of the benchmark. Default is 3.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if not self._set_binary_path():
            return False

        # Construct the command for nvbandwidth
        command = os.path.join(self._args.bin_dir, self._bin_name)

        if self._args.buffer_size:
            command += f' --bufferSize {self._args.buffer_size}'

        if self._args.test_cases:
            command += ' --testcase ' + ' '.join([testcase.strip() for testcase in self._args.test_cases.split(',')])

        if self._args.skip_verification:
            command += ' --skipVerification'

        if self._args.disable_affinity:
            command += ' --disableAffinity'

        if self._args.use_mean:
            command += ' --useMean'

        if self._args.num_loops:
            command += f' --testSamples {self._args.num_loops}'

        self._commands.append(command)

        return True

    def _process_raw_line(self, line, parse_status):
        """Process a single line of raw output from the nvbandwidth benchmark.

        This function updates the `parse_status` dictionary with parsed results from the given `line`.
        It detects the start of a test, parses matrix headers and rows, and extracts summary results.

        Args:
            line (str): A single line of raw output from the benchmark.
            parse_status (dict): A dictionary to maintain the current parsing state and results. It should contain:
                - 'test_name' (str): The name of the current test being parsed.
                - 'benchmark_type' (str): 'bw' or 'lat'. It also indicating if matrix data is being parsed.
                - 'matrix_header' (list): The header of the matrix being parsed.
                - 'results' (dict): A dictionary to store the parsed results.

        Return:
            None
        """
        # Regular expressions for summary line and matrix header detection
        block_start_pattern = re.compile(r'^Running\s+(.+)$')
        summary_pattern = re.compile(r'SUM (\S+) (\d+\.\d+)')
        matrix_header_line = re.compile(r'^(memcpy|memory latency)')
        matrix_row_pattern = re.compile(r'^\s*\d')

        line = line.strip()

        # Detect the start of a test
        if block_start_pattern.match(line):
            parse_status['test_name'] = block_start_pattern.match(line).group(1).lower()[:-1]
            return

        # Detect the start of matrix data
        if parse_status['test_name'] and matrix_header_line.match(line):
            parse_status['benchmark_type'] = 'bw' if 'bandwidth' in line else 'lat'
            return

        # Parse the matrix header
        if (
            parse_status['test_name'] and parse_status['benchmark_type'] and not parse_status['matrix_header']
            and matrix_row_pattern.match(line)
        ):
            parse_status['matrix_header'] = line.split()
            return

        # Parse matrix rows
        if parse_status['test_name'] and parse_status['benchmark_type'] and matrix_row_pattern.match(line):
            row_data = line.split()
            row_index = row_data[0]
            for col_index, value in enumerate(row_data[1:], start=1):
                col_header = parse_status['matrix_header'][col_index - 1]
                test_name = parse_status['test_name']
                benchmark_type = parse_status['benchmark_type']
                metric_name = f'{test_name}_cpu{row_index}_gpu{col_header}_{benchmark_type}'
                parse_status['results'][metric_name] = float(value)
            return

        # Parse summary results
        summary_match = summary_pattern.search(line)
        if summary_match:
            value = float(summary_match.group(2))
            test_name = parse_status['test_name']
            benchmark_type = parse_status['benchmark_type']
            parse_status['results'][f'{test_name}_sum_{benchmark_type}'] = value

            # Reset parsing state for next test
            parse_status['test_name'] = ''
            parse_status['benchmark_type'] = None
            parse_status['matrix_header'].clear()

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
            self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)
            content = raw_output.splitlines()
            parsing_status = {
                'results': {},
                'benchmark_type': None,
                'matrix_header': [],
                'test_name': '',
            }

            for line in content:
                self._process_raw_line(line, parsing_status)

            if not parsing_status['results']:
                self._result.add_raw_data('nvbandwidth', 'No valid results found', self._args.log_raw_data)
                return False

            # Store parsed results
            for metric, value in parsing_status['results'].items():
                self._result.add_result(metric, value)

            return True
        except Exception as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            self._result.add_result('abort', 1)
            return False


BenchmarkRegistry.register_benchmark('nvbandwidth', NvBandwidthBenchmark, platform=Platform.CUDA)
