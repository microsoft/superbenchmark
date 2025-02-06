# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the NV Bandwidth Test."""

import os
import subprocess
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class NvBandwidthBenchmark(MicroBenchmarkWithInvoke):
    """The NV Bandwidth Test benchmark class."""
    # Regular expressions for summary line and matrix header detection
    re_block_start_pattern = re.compile(r'^Running\s+(.+)$')
    re_matrix_header_line = re.compile(r'^(memcpy|memory latency)')
    re_matrix_row_pattern = re.compile(r'^\s*\d')
    re_summary_pattern = re.compile(r'SUM (\S+) (\d+\.\d+)')
    re_unsupported_pattern = re.compile(r'ERROR: Testcase (\S+) not found!')

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
            nargs='+',
            type=str,
            default=[],
            required=False,
            help=(
                'Specify the test case(s) to execute by name only. '
                'To view the available test case names, run the command "nvbandwidth -l" on the host. '
                'If no specific test case is specified, all test cases will be executed by default.'
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
            command += ' --testcase ' + ' '.join(self._args.test_cases)
        else:
            self._args.test_cases = self._get_all_test_cases()

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
        """Process a raw line of text and update the parse status accordingly.

        Args:
            line (str): The raw line of text to be processed.
            parse_status (dict): A dictionary containing the current parsing status,
                     which will be updated based on the content of the line.

        Returns:
            None
        """
        line = line.strip()

        # Detect unsupported test cases
        if self.re_unsupported_pattern.match(line):
            parse_status['unsupported_testcases'].add(self.re_unsupported_pattern.match(line).group(1).lower())
            return

        # Detect the start of a test
        if self.re_block_start_pattern.match(line):
            parse_status['test_name'] = self.re_block_start_pattern.match(line).group(1).lower()[:-1]
            parse_status['excuted_testcases'].add(parse_status['test_name'])
            return

        # Detect the start of matrix data
        if parse_status['test_name'] and self.re_matrix_header_line.match(line):
            parse_status['benchmark_type'] = 'bw' if 'bandwidth' in line else 'lat'
            # Parse the row and column name
            tmp_idx = line.find('(row)')
            parse_status['metrix_row'] = line[tmp_idx - 3:tmp_idx].lower()
            tmp_idx = line.find('(column)')
            parse_status['metrix_col'] = line[tmp_idx - 3:tmp_idx].lower()
            return

        # Parse the matrix header
        if (
            parse_status['test_name'] and parse_status['benchmark_type'] and not parse_status['matrix_header']
            and self.re_matrix_row_pattern.match(line)
        ):
            parse_status['matrix_header'] = line.split()
            return

        # Parse matrix rows
        if parse_status['test_name'] and parse_status['benchmark_type'] and self.re_matrix_row_pattern.match(line):
            row_data = line.split()
            row_index = row_data[0]
            for col_index, value in enumerate(row_data[1:], start=1):
                # Skip 'N/A' values, 'N/A' indicates the test path is self to self.
                if value == 'N/A':
                    continue

                col_header = parse_status['matrix_header'][col_index - 1]
                test_name = parse_status['test_name']
                benchmark_type = parse_status['benchmark_type']
                row_name = parse_status['metrix_row']
                col_name = parse_status['metrix_col']
                metric_name = f'{test_name}_{row_name}{row_index}_{col_name}{col_header}_{benchmark_type}'
                parse_status['results'][metric_name] = float(value)
            return

        # Parse summary results
        if self.re_summary_pattern.match(line):
            value = self.re_summary_pattern.match(line).group(2)
            test_name = parse_status['test_name']
            benchmark_type = parse_status['benchmark_type']
            parse_status['results'][f'{test_name}_sum_{benchmark_type}'] = float(value)

            # Reset parsing state for next test
            parse_status['test_name'] = ''
            parse_status['benchmark_type'] = None
            parse_status['matrix_header'].clear()
            parse_status['metrix_row'] = ''
            parse_status['metrix_col'] = ''
            return

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
                'excuted_testcases': set(),
                'unsupported_testcases': set(),
                'benchmark_type': None,
                'matrix_header': [],
                'test_name': '',
                'metrix_row': '',
                'metrix_col': '',
            }

            for line in content:
                self._process_raw_line(line, parsing_status)

            return_code = ReturnCode.SUCCESS
            # Log unsupported test cases
            for testcase in parsing_status['unsupported_testcases']:
                logger.warning(f'Test case {testcase} is not supported.')
                return_code = ReturnCode.INVALID_ARGUMENT
                self._result.add_raw_data(testcase, 'Not supported', self._args.log_raw_data)

            # Check if the test case was waived
            for testcase in self._args.test_cases:
                if (
                    testcase not in parsing_status['unsupported_testcases']
                    and testcase not in parsing_status['excuted_testcases']
                ):
                    logger.warning(f'Test case {testcase} was waived.')
                    self._result.add_raw_data(testcase, 'waived', self._args.log_raw_data)
                    return_code = ReturnCode.INVALID_ARGUMENT

            if not parsing_status['results']:
                self._result.add_raw_data('nvbandwidth', 'No valid results found', self._args.log_raw_data)
                return_code = ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE
                return False

            # Store parsed results
            for metric, value in parsing_status['results'].items():
                self._result.add_result(metric, value)

            self._result.set_return_code(return_code)
            return True
        except Exception as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            self._result.add_result('abort', 1)
            return False

    def _get_all_test_cases(self):
        command = os.path.join(self._args.bin_dir, self._bin_name) + ' --list'
        test_case_pattern = re.compile(r'(\d+),\s+([\w_]+):')

        try:
            # Execute the command and capture output
            result = subprocess.run(
                command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )

            # Check the return code
            if result.returncode != 0:
                logger.error(f'{command} failed with return code {result.returncode}')
                return []

            if result.stderr:
                logger.error(f'{command} failed with {result.stderr}')
                return []

            # Parse the output
            return [name for _, name in test_case_pattern.findall(result.stdout)]
        except Exception as e:
            logger.error(f'Failed to get all test case names: {e}')
            return []


BenchmarkRegistry.register_benchmark('nvbandwidth', NvBandwidthBenchmark, platform=Platform.CUDA)
