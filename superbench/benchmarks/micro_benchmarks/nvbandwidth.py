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

        self._commands.append(command)

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
        try:
            self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)
            content = raw_output.splitlines()
            results = {}
            parsing_matrix = False
            matrix_header = []
            test_name = ""

            # Regular expressions for summary line and matrix header detection
            block_start_pattern = re.compile(r"^Running\s+(.+)$")
            summary_pattern = re.compile(r'SUM (\S+) (\d+\.\d+)')
            matrix_header_line = re.compile(r'^memcpy CE CPU\(row\)')
            matrix_row_pattern = re.compile(r'^\s*\d')

            for line in content:
                line = line.strip()

                # Detect the start of a test
                if block_start_pattern.match(line):
                    test_name = block_start_pattern.match(line).group(1).lower()[:-1]
                    continue

                # Detect the start of matrix data
                if test_name and matrix_header_line.match(line):
                    parsing_matrix = True
                    continue

                # Parse the matrix header
                if test_name and parsing_matrix and not matrix_header and matrix_row_pattern.match(line):
                    matrix_header = line.split()
                    continue

                # Parse matrix rows
                if test_name and parsing_matrix and matrix_row_pattern.match(line):
                    row_data = line.split()
                    row_index = row_data[0]
                    for col_index, value in enumerate(row_data[1:], start=1):
                        col_header = matrix_header[col_index - 1]
                        metric_name = f'{test_name}_bandwidth_cpu{row_index}_gpu{col_header}'
                        results[metric_name] = float(value)
                    continue

                # Parse summary results
                summary_match = summary_pattern.search(line)
                if summary_match:
                    value = float(summary_match.group(2))
                    results[f'{test_name}_sum_bandwidth'] = value

                    # Reset parsing state for next test
                    test_name = ""
                    parsing_matrix = False
                    matrix_header.clear()

            if not results:
                self._result.add_raw_data('nvbandwidth', 'No valid results found', self._args.log_raw_data)
                return False

            # Store parsed results
            for metric, value in results.items():
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
