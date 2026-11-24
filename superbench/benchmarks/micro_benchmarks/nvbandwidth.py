# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the NV Bandwidth Test."""

import os
import subprocess
import re

from superbench.common.utils import logger, run_command
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class NvBandwidthBenchmark(MicroBenchmarkWithInvoke):
    """The NV Bandwidth Test benchmark class."""
    # Regular expressions for summary line and matrix header detection
    re_version_pattern = re.compile(r'^nvbandwidth Version:\s*v?(\d+)\.(\d+)', re.MULTILINE)
    re_block_start_pattern = re.compile(r'^Running\s+(\w+)\.$')
    re_matrix_header_line = re.compile(
        r'^(memcpy|memory latency|Device to Device Latency|(?:UBLKCP|fabric) device-to-device)\b', re.IGNORECASE
    )
    re_matrix_axis_pattern = re.compile(r'\b(CPU|GPU)\((row|column)\)', re.IGNORECASE)
    # v0.11 UBLKCP/fabric latency titles label both axes at once, e.g. GPU(row->col).
    re_matrix_pair_axis_pattern = re.compile(r'\b(CPU|GPU)\(row->col\)', re.IGNORECASE)
    re_matrix_direction_pattern = re.compile(r'\b(Read[12]|Write[12]|Total) bandwidth\b', re.IGNORECASE)
    re_matrix_row_pattern = re.compile(r'^\s*\d')
    re_summary_pattern = re.compile(r'^SUM\s+(\S+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$')
    re_unsupported_pattern = re.compile(r'^(?:ERROR: Testcase (\S+) not found!|ERROR (\S+): Testcase \S+ not found!)$')
    re_status_pattern = re.compile(r'^(Waived|Error)(?:\s+(\w+))?:\s*(.*)$', re.IGNORECASE)

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
            help=(
                'Disable automatic CPU affinity control. Default is False. '
                'If user would like to bind the process to specific NUMA node, '
                'please use --disable_affinity along with --numa argument.'
            ),
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

        if not self._set_binary_path() or not self._get_arguments_from_env():
            return False

        # Construct the command for nvbandwidth
        command = os.path.join(self._args.bin_dir, self._bin_name)

        if self._args.buffer_size:
            command += f' --bufferSize {self._args.buffer_size}'

        if self._args.test_cases:
            command += ' --testcase ' + ' '.join(self._args.test_cases)
        else:
            self._args.test_cases = self._get_all_test_cases()
            if not self._args.test_cases:
                logger.error('Cannot run nvbandwidth without discovered testcases.')
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
                return False

        for enabled, flag in (
            (self._args.skip_verification, '--skipVerification'),
            (self._args.disable_affinity, '--disableAffinity'),
            (self._args.use_mean, '--useMean'),
        ):
            if enabled:
                command += ' ' + flag

        if self._args.disable_affinity and self._args.numa is not None:
            command = f'numactl --cpunodebind={self._args.numa} --membind={self._args.numa} ' + command

        if self._args.num_loops:
            command += f' --testSamples {self._args.num_loops}'

        self._commands.append(command)

        return True

    def _process_matrix_line(self, line, parse_status):
        """Parse a single-node matrix header or row without ending its enclosing testcase.

        Args:
            line (str): Stripped output line.
            parse_status (dict): Current testcase, matrix state and collected results.
        """
        test_name = parse_status['test_name']
        if not test_name:
            return

        if self.re_matrix_header_line.match(line):
            axes = {axis.lower(): name.lower() for name, axis in self.re_matrix_axis_pattern.findall(line)}
            pair_axis = self.re_matrix_pair_axis_pattern.search(line)
            if pair_axis:
                axes = {'row': pair_axis.group(1).lower(), 'column': pair_axis.group(1).lower()}
            direction_match = self.re_matrix_direction_pattern.search(line)
            suffix = direction_match.group(1).lower() if direction_match else ''
            parse_status['matrix_index'] += 1
            parse_status['matrix_name'] = test_name + ('_' + suffix if suffix else '')
            parse_status['benchmark_type'] = 'bw' if 'bandwidth' in line.lower() else 'lat'
            parse_status['matrix_header'] = []
            parse_status['metrix_row'] = axes.get('row', '')
            parse_status['metrix_col'] = axes.get('column', '')
            parse_status['matrix_has_values'] = False
            expected = parse_status['testcase_matrices'][test_name]
            if suffix:
                direction = 'read' if '_read_' in test_name else 'write'
                expected.update((f'{test_name}_{direction}1', f'{test_name}_{direction}2', f'{test_name}_total'))
            else:
                expected.add(test_name)
            return

        if not parse_status['benchmark_type'] or not self.re_matrix_row_pattern.match(line):
            return

        row_data = line.split()
        if not parse_status['matrix_header']:
            parse_status['matrix_header'] = row_data
            return

        for col_index, value in enumerate(row_data[1:]):
            if value == 'N/A':
                continue
            row_name = parse_status['metrix_row']
            col_name = parse_status['metrix_col']
            col_header = parse_status['matrix_header'][col_index]
            row_metric = f'{row_name}{row_data[0]}_' if row_name else ''
            metric = f'{row_metric}{col_name}{col_header}_{parse_status["benchmark_type"]}'
            measurement = float(value)
            parse_status['results'].setdefault(f'{parse_status["matrix_name"]}_{metric}', measurement)
            if parse_status['matrix_index'] == 1:
                parse_status['results'].setdefault(f'{test_name}_{metric}', measurement)
            parse_status['matrix_has_values'] = True

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

        unsupported = self.re_unsupported_pattern.match(line)
        if unsupported:
            parse_status['unsupported_testcases'].add((unsupported.group(1) or unsupported.group(2)).lower())
            return

        running = self.re_block_start_pattern.match(line)
        if running:
            test_name = running.group(1).lower()
            parse_status.update(
                test_name=test_name,
                matrix_name='',
                matrix_index=0,
                benchmark_type=None,
                matrix_header=[],
                matrix_has_values=False,
            )
            parse_status['testcase_matrices'].setdefault(test_name, set())
            return

        status = self.re_status_pattern.match(line)
        if status:
            state, named_test, message = status.groups()
            test_name = (named_test or parse_status['test_name']).lower()
            completed = bool(
                parse_status['testcase_matrices'].get(test_name, set()) & parse_status['completed_matrices']
            )
            # Legacy output has anonymous, empty waivers between completed testcases.
            legacy_waiver = (
                parse_status['legacy_waivers'] and state.lower() == 'waived' and not named_test and not message
                and completed
            )
            if not test_name or legacy_waiver:
                return
            status_key = 'waived_testcases' if state.lower() == 'waived' else 'failed_testcases'
            parse_status[status_key][test_name] = message
            parse_status.update(test_name='', matrix_name='', benchmark_type=None, matrix_header=[])
            return

        summary = self.re_summary_pattern.match(line)
        if summary:
            if not parse_status['matrix_name']:
                return
            matrix_name, value = summary.groups()
            matrix_name = matrix_name.lower()
            test_name = parse_status['test_name']
            benchmark_type = parse_status['benchmark_type']
            if matrix_name != parse_status['matrix_name']:
                parse_status['failed_testcases'][test_name] = f'Unexpected SUM identifier: {matrix_name}'
            elif parse_status['matrix_has_values']:
                parse_status['results'][f'{matrix_name}_sum_{benchmark_type}'] = float(value)
                if parse_status['matrix_index'] == 1:
                    parse_status['results'].setdefault(f'{test_name}_sum_{benchmark_type}', float(value))
                parse_status['completed_matrices'].add(matrix_name)
            # A SUM ends a matrix, not necessarily its enclosing testcase.
            parse_status.update(matrix_name='', benchmark_type=None, matrix_header=[])
            return

        self._process_matrix_line(line, parse_status)

    def _get_testcase_return_code(self, parsing_status):
        """Report unsupported, incomplete and failed testcases.

        Args:
            parsing_status (dict): Parsed testcase outcomes and completed matrices.

        Return:
            ReturnCode: Combined testcase status.
        """
        return_code = ReturnCode.SUCCESS
        for testcase in parsing_status['unsupported_testcases']:
            logger.warning(f'Test case {testcase} is not supported.')
            return_code = ReturnCode.INVALID_ARGUMENT
            self._result.add_raw_data(testcase, 'Not supported', self._args.log_raw_data)

        # Running alone, or only one direction of a bidirectional test, is not success.
        waived = set(parsing_status['waived_testcases'])
        testcases = {name.lower() for name in self._args.test_cases} | set(parsing_status['testcase_matrices'])
        for testcase in testcases:
            expected = parsing_status['testcase_matrices'].get(testcase, set())
            if not expected or not expected <= parsing_status['completed_matrices']:
                waived.add(testcase)
        waived -= parsing_status['unsupported_testcases'] | set(parsing_status['failed_testcases'])
        for testcase in waived:
            reason = parsing_status['waived_testcases'].get(testcase) or 'No complete results'
            logger.warning(f'Test case {testcase} was waived or incomplete: {reason}.')
            self._result.add_raw_data(testcase, 'waived', self._args.log_raw_data)
            return_code = ReturnCode.INVALID_ARGUMENT

        for testcase, message in parsing_status['failed_testcases'].items():
            logger.error(f'Test case {testcase} failed: {message}.')
            self._result.add_raw_data(testcase, f'Error: {message}', self._args.log_raw_data)
            return_code = ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE

        return return_code

    def _benchmark(self):
        """Run nvbandwidth and parse its output even after a nonzero exit.

        nvbandwidth aborts on a later CUDA error after printing earlier complete matrices, so those
        measurements are kept while the run still reports MICROBENCHMARK_EXECUTION_FAILURE. Return codes
        set by the parser, including explicit testcase outcomes, are not replaced by a generic parsing failure.

        Return:
            True if every command exited successfully and produced parsable results.
        """
        ret = True
        for cmd_idx, command in enumerate(self._commands):
            logger.info(
                f'Execute command - round: {self._curr_run_index}, benchmark: {self._name}, command: {command}.'
            )
            output = run_command(command, flush_output=self._args.log_flushing, cwd=self._args.bin_dir)
            parsed = self._process_raw_result(cmd_idx, output.stdout)
            if output.returncode != 0:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
                logger.error(
                    f'Microbenchmark execution failed - round: {self._curr_run_index}, benchmark: {self._name}, '
                    f'return code: {output.returncode}.'
                )
            if output.returncode != 0 or not parsed:
                ret = False
                if not self._args.tolerant_fail:
                    return False
        return ret

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
            version = self.re_version_pattern.search(raw_output)
            parsing_status = {
                'results': {},
                'legacy_waivers': bool(version and tuple(map(int, version.groups())) < (0, 11)),
                'unsupported_testcases': set(),
                'waived_testcases': {},
                'failed_testcases': {},
                'testcase_matrices': {},
                'completed_matrices': set(),
                'benchmark_type': None,
                'matrix_header': [],
                'matrix_name': '',
                'matrix_index': 0,
                'matrix_has_values': False,
                'test_name': '',
                'metrix_row': '',
                'metrix_col': '',
            }

            for line in content:
                self._process_raw_line(line, parsing_status)

            return_code = self._get_testcase_return_code(parsing_status)

            if not parsing_status['results']:
                self._result.add_raw_data('nvbandwidth', 'No valid results found', self._args.log_raw_data)
                explicit_outcome = any(
                    parsing_status[key] for key in ('unsupported_testcases', 'waived_testcases', 'failed_testcases')
                )
                self._result.set_return_code(
                    return_code if explicit_outcome else ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE
                )
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
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
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
            test_cases = [name for _, name in test_case_pattern.findall(result.stdout)]
            if not test_cases:
                logger.error(f'{command} returned no testcases.')
            return test_cases
        except Exception as e:
            logger.error(f'Failed to get all test case names: {e}')
            return []


BenchmarkRegistry.register_benchmark('nvbandwidth', NvBandwidthBenchmark, platform=Platform.CUDA)
