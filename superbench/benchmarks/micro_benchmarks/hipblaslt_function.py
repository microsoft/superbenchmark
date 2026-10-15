# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the hipBlasLt GEMM benchmark."""

import math
import os
import re
import shlex
from typing import Dict, Tuple

from superbench.common.utils import LazyImport, logger, run_command
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import BlasLtBaseBenchmark

torch = LazyImport('torch')


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
        self._in_types = ['fp32', 'fp16', 'bf16', 'fp8', 'mxfp4']
        self._in_type_map = {
            'fp16': '--a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r',
            'fp32': '--a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r',
            'bf16': '--a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r',
            'fp8': '--a_type f8_r --b_type f8_r --c_type f8_r --d_type f8_r --compute_type f32_r',
            'mxfp4': '--a_type f4_r --b_type f4_r --c_type f32_r --d_type f32_r --compute_type f32_r'
            ' --scaleA 3 --scaleB 3 --scaleABlockRowSize 32 --scaleABlockColSize 1'
            ' --scaleBBlockRowSize 1 --scaleBBlockColSize 32 --alpha 1 --beta 0'
            ' --algo_method heuristic --requested_solution 1 --verify 1 --use_gpu_timer 1 --print_kernel_info 1',
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
        self._parser.add_argument(
            '--enable_experimental_fp4',
            action='store_true',
            help='Opt in to unvalidated MXFP4 on gfx950 with a block-scaling hipblaslt-bench client.',
        )

    def _validate_mxfp4_arguments(self) -> None:
        """Validate the experimental non-batched, TN MXFP4 case contract."""
        if not self._args.enable_experimental_fp4:
            raise ValueError('MXFP4 requires --enable_experimental_fp4; hardware validation is pending.')
        if (self._args.transA, self._args.transB) != ('T', 'N'):
            raise ValueError('MXFP4 currently requires --transA T --transB N.')
        if self._args.initialization not in ['hpl', 'trig_float']:
            raise ValueError('MXFP4 requires --initialization hpl or trig_float.')
        if self._args.num_warmup < 0 or self._args.num_steps < 1:
            raise ValueError('MXFP4 requires non-negative warmup and positive measurement iterations.')
        for m, n, k, batch, precision in self._shapes_to_run:
            if precision == 'mxfp4' and (batch != 0 or min(m, n, k) <= 0 or k % 32 != 0):
                raise ValueError('MXFP4 requires --batch 0, positive dimensions, and K divisible by 32.')

    def _check_mxfp4_support(self) -> bool:
        """Check the selected ROCm device and the client's block-scaling options."""
        try:
            if torch.version.hip is None or not torch.cuda.is_available():
                raise RuntimeError('MXFP4 requires a working ROCm PyTorch device.')
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
        except (AttributeError, RuntimeError) as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
            logger.error('Cannot identify the MXFP4 device - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        if arch != 'gfx950':
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_UNSUPPORTED_ARCHITECTURE)
            logger.error('Experimental MXFP4 only allows gfx950 - benchmark: {}, architecture: {}.'.format(
                self._name, arch
            ))
            return False

        output = run_command(f'{shlex.quote(self.__bin_path)} --help', quiet=True)
        self._result.add_raw_data('mxfp4_client_help', output.stdout, self._args.log_raw_data)
        if output.returncode != 0:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            logger.error(
                'Cannot query MXFP4 client options - benchmark: {}, output: {}.'.format(self._name, output.stdout)
            )
            return False
        required_options = [
            '--scaleA', '--scaleB', '--scaleABlockRowSize', '--scaleABlockColSize',
            '--scaleBBlockRowSize', '--scaleBBlockColSize', '--verify', '--use_gpu_timer',
            '--print_kernel_info', '3 = block',
        ]
        missing = [option for option in required_options if not re.search(re.escape(option) + r'\b', output.stdout)]
        if missing:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('MXFP4 client is missing required options - benchmark: {}, missing: {}.'.format(
                self._name, missing
            ))
            return False
        self._result.add_raw_data('mxfp4_gpu_arch', arch, self._args.log_raw_data)
        return True

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
        if 'mxfp4' in self._args.in_types:
            try:
                self._validate_mxfp4_arguments()
            except ValueError as e:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('Invalid MXFP4 arguments - benchmark: {}, message: {}.'.format(self._name, str(e)))
                return False
            if not self._check_mxfp4_support():
                return False

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

    def _parse_result_row(self, raw_output: str, precision: str) -> Dict[str, str]:
        """Parse one result using header names, preserving legacy column handling."""
        lines = raw_output.splitlines()
        index = None
        for i, line in enumerate(lines):
            if 'hipblaslt-Gflops' in line:
                index = i
                break
        if index is None:
            raise ValueError('Line with "hipblaslt-Gflops" not found in the log.')

        header_fields = [field.strip() for field in lines[index].split(',')]
        header_fields[0] = re.sub(r'^\s*\[\d+\]:?', '', header_fields[0]).strip()
        if precision == 'mxfp4' and (
            len(set(header_fields)) != len(header_fields)
            or sum('hipblaslt-Gflops' in line for line in lines) != 1
            or any(len(line.split(',')) == len(header_fields) for line in lines[index + 2:])
        ):
            raise ValueError('MXFP4 requires unique columns and exactly one result.')

        col_idx_by_name: Dict[str, int] = {}
        for col_idx, col_name in enumerate(header_fields):
            col_idx_by_name.setdefault(col_name, col_idx)
        required_columns = ['batch_count', 'm', 'n', 'k', 'hipblaslt-Gflops']
        missing_columns = [column for column in required_columns if column not in col_idx_by_name]
        if missing_columns:
            raise ValueError(f'Required column(s) not found in header: {missing_columns}.')
        if index + 1 >= len(lines):
            raise ValueError('Data line missing after "hipblaslt-Gflops" header.')

        fields = [field.strip() for field in lines[index + 1].split(',')]
        if len(fields) != len(header_fields):
            raise ValueError(
                f'Field count mismatch: header has {len(header_fields)} columns but data has {len(fields)} columns'
            )
        return {column: fields[col_idx] for column, col_idx in col_idx_by_name.items()}

    def _validate_mxfp4_result(self, cmd_idx: int, row: Dict[str, str], gflops: float) -> Tuple[float, float]:
        """Require the requested MXFP4 case, valid timing, and CPU verification."""
        m, n, k, _, _ = self._shapes_to_run[cmd_idx]
        expected = {
            'transA': 'T', 'transB': 'N', 'grouped_gemm': '0', 'batch_count': '1',
            'm': str(m), 'n': str(n), 'k': str(k), 'a_type': 'f4_r', 'b_type': 'f4_r',
            'c_type': 'f32_r', 'd_type': 'f32_r', 'compute_type': 'f32_r',
            'scaleA': '3', 'scaleB': '3', 'use_gpu_timer': '1',
        }
        for column, value in expected.items():
            if row.get(column) != value:
                raise ValueError(f'MXFP4 column {column} must be {value}, got {row.get(column)}.')

        latency = float(row['us'])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError('MXFP4 latency must be finite and positive.')
        if not math.isclose(gflops, 2 * m * n * k / latency / 1000, rel_tol=0.01):
            raise ValueError('MXFP4 throughput does not match 2*M*N*K and the reported latency.')

        # hipblaslt-bench reports verification errors without asserting them.
        norm_error = float(row['norm_error'])
        if not 0 <= norm_error < 1e-5:
            raise ValueError('MXFP4 FP32-output relative Frobenius norm error must be finite and below 1e-5.')
        for column in ['atol', 'rtol']:
            error = float(row[column])
            if not math.isfinite(error) or error < 0:
                raise ValueError(f'MXFP4 verification column {column} must be finite and non-negative.')
        return latency, norm_error

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
            precision = self._precision_in_commands[cmd_idx]
            row = self._parse_result_row(raw_output, precision)
            gflops = float(row['hipblaslt-Gflops'])
            if not math.isfinite(gflops) or gflops <= 0:
                raise ValueError('Throughput must be finite and positive.')
            metric = f'{precision}_{row["batch_count"]}_{row["m"]}_{row["n"]}_{row["k"]}'
            if precision == 'mxfp4':
                latency, norm_error = self._validate_mxfp4_result(cmd_idx, row, gflops)
                self._result.add_result(f'{metric}_time', latency)
                self._result.add_result(f'{metric}_norm_error', norm_error)
                self._result.add_result(f'{metric}_correctness', 1)
            self._result.add_result(f'{metric}_flops', gflops / 1000)
        except (IndexError, KeyError, TypeError, ValueError) as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('hipblaslt-gemm', HipBlasLtBenchmark, platform=Platform.ROCM)
