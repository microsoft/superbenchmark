# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the hipBlasLt GEMM benchmark."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import BlasLtBaseBenchmark


class HipBlasLtBenchmark(BlasLtBaseBenchmark):
    """The hipBlasLt GEMM benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'hipblaslt-bench'
        self._in_types = ['fp32', 'fp16', 'bf16', 'fp8']
        self._in_type_map = {
            'fp16': '--a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r',
            'fp32': '--a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r',
            'bf16': '--a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r',
            'fp8': '--a_type f8_r --b_type f8_r --c_type f8_r --d_type f8_r --compute_type f32_r',
        }

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--in_types',
            type=str,
            nargs='+',
            default=['fp16'],
            required=False,
            help='List of input data types, support {}.'.format(' '.join(self._in_types)),
        )
        self._parser.add_argument(
            '--initialization',
            type=str,
            default='rand_int',
            choices=['trig_float', 'rand_int', 'hpl'],
            required=False,
            help='Initialize matrix data.',
        )
        self._parser.add_argument(
            '--transA',
            type=str,
            default='N',
            choices=['N', 'T', 'C'],
            required=False,
            help='Transpose matrix A.',
        )
        self._parser.add_argument(
            '--transB',
            type=str,
            default='N',
            choices=['N', 'T', 'C'],
            required=False,
            help='Transpose matrix B.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        self._commands = []
        self._precision_in_commands = []
        for (_m, _n, _k, _b, _in_type) in self._shapes_to_run:
            command = f'{self.__bin_path} -m {_m} -n {_n} -k {_k} -j {self._args.num_warmup}' + \
                f' -i {self._args.num_steps} {self._in_type_map[_in_type]}' + \
                f' --transA {self._args.transA} --transB {self._args.transB}' + \
                f' --initialization {self._args.initialization}'
            command = command + f' -b {str(_b)}' if _b > 0 else command
            logger.info(command)
            self._commands.append(command)
            self._precision_in_commands.append(_in_type)

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
        self._result.add_raw_data(f'raw_output_{cmd_idx}', raw_output, self._args.log_raw_data)

        try:
            lines = raw_output.splitlines()
            index = None

            # Find the header line containing 'hipblaslt-Gflops'
            for i, line in enumerate(lines):
                if 'hipblaslt-Gflops' in line:
                    index = i
                    break

            if index is None:
                raise ValueError('Line with "hipblaslt-Gflops" not found in the log.')

            # Parse the header and resolve every key column (batch_count/m/n/k/hipblaslt-Gflops)
            # by name. This keeps the parser forward-compatible across known and future
            # hipBLASLt output formats (v600: 23 columns; v1500: 34 columns with extra
            # a_type/b_type/c_type/scaleA-D/amaxD/bias_type/aux_type/hipblaslt-GB/s),
            # without relying on any fixed column position.
            header_fields = lines[index].strip().split(',')
            # Strip leading rank markers like '[0]' or '[0]:' from the first header field.
            # Use a regex anchored at the start so a column name that legitimately contains
            # ']' (unlikely, but defensive) is not truncated.
            header_fields[0] = re.sub(r'^\s*\[\d+\]:?', '', header_fields[0])

            # Build a name -> column-index map (first occurrence wins for any duplicates).
            col_idx_by_name = {}
            for col_idx, col_name in enumerate(header_fields):
                col_idx_by_name.setdefault(col_name.strip(), col_idx)

            required_columns = ['batch_count', 'm', 'n', 'k', 'hipblaslt-Gflops']
            missing_columns = [c for c in required_columns if c not in col_idx_by_name]
            if missing_columns:
                raise ValueError(f'Required column(s) not found in header: {missing_columns}.')

            # Ensure a data line follows the header (e.g., hipblaslt-bench may have
            # crashed after printing the header).
            if index + 1 >= len(lines):
                raise ValueError('Data line missing after "hipblaslt-Gflops" header.')

            # Split the data line into fields using a comma as the delimiter
            fields = lines[index + 1].strip().split(',')

            # Validate that the data line has the same number of columns as the header
            if len(fields) != len(header_fields):
                raise ValueError(
                    f'Field count mismatch: header has {len(header_fields)} columns '
                    f'but data has {len(fields)} columns'
                )

            # Resolve every key value by header name and strip whitespace from each, so
            # any padding around CSV values does not bleed into the metric key.
            batch_count = fields[col_idx_by_name['batch_count']].strip()
            m_val = fields[col_idx_by_name['m']].strip()
            n_val = fields[col_idx_by_name['n']].strip()
            k_val = fields[col_idx_by_name['k']].strip()
            gflops_col = col_idx_by_name['hipblaslt-Gflops']

            self._result.add_result(
                f'{self._precision_in_commands[cmd_idx]}_{batch_count}_{m_val}_{n_val}_{k_val}_flops',
                float(fields[gflops_col]) / 1000
            )
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('hipblaslt-gemm', HipBlasLtBenchmark, platform=Platform.ROCM)
