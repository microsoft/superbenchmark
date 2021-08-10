# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FLOPs performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class RocmGemmFlops(MicroBenchmarkWithInvoke):
    """The GEMM FLOPs performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'rocblas-bench'

        self.__precision_map = {
            'FP64': '-r f64_r -f gemm',
            'FP32_X': '-r f32_r -f gemm_ex --compute_type f32_r',
            'FP16_X': '-r f16_r -f gemm_ex --compute_type f32_r',
            'BF16_X': '-r bf16_r -f gemm_ex --compute_type f32_r',
            'INT8_X': '--a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r -f gemm_ex --compute_type i32_r'
        }

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--precision',
            type=str,
            nargs='+',
            default=None,
            help='Precision for benchmarking. E.g. {}.'.format(' '.join(list(self.__precision_map.keys()))),
        )
        self._parser.add_argument(
            '--transposeA',
            type=str,
            default='N',
            help='Transpose type of Matrix A, N = no transpose, T = transpose, C = conjugate transpose',
        )
        self._parser.add_argument(
            '--transposeB',
            type=str,
            default='T',
            help='Transpose type of Matrix B, N = no transpose, T = transpose, C = conjugate transpose',
        )
        self._parser.add_argument(
            '--m',
            type=int,
            default=7680,
            required=False,
            help='The M dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--n',
            type=int,
            default=8192,
            required=False,
            help='The N dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--k',
            type=int,
            default=8192,
            required=False,
            help='The K dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--lda',
            type=int,
            default=8384,
            required=False,
            help='Leading dimension of matrix A.',
        )
        self._parser.add_argument(
            '--ldb',
            type=int,
            default=8384,
            required=False,
            help='Leading dimension of matrix B.',
        )
        self._parser.add_argument(
            '--ldc',
            type=int,
            default=8384,
            required=False,
            help='Leading dimension of matrix C.',
        )
        self._parser.add_argument(
            '--ldd',
            type=int,
            default=8384,
            required=False,
            help='Leading dimension of matrix D.',
        )
        self._parser.add_argument(
            '--alpha',
            type=int,
            default=1,
            required=False,
            help='Specifies the scalar alpha.',
        )
        self._parser.add_argument(
            '--beta',
            type=int,
            default=0,
            required=False,
            help='Specifies the scalar beta.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if self._args.precision is None:
            self._args.precision = list(self.__precision_map.keys())
        else:
            self._args.precision = [p.upper() for p in self._args.precision]
        for p in self._args.precision:
            if p not in self.__precision_map:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.warning(
                    'Unsupported precision - benchmark: {}, precision: {}, expected: {}.'.format(
                        self._name, p, list(self.__precision_map.keys())
                    )
                )
                return False
            else:
                command = os.path.join(self._args.bin_dir, self._bin_name)
                command += ' ' + self.__precision_map[p]
                command += ' --transposeA {} --transposeB {}'.format(self._args.transposeA, self._args.transposeB)
                command += ' -m {} -n {} -k {}'.format(self._args.m, self._args.n, self._args.k)
                command += ' --alpha {} --beta {}'.format(self._args.alpha, self._args.beta)
                command += ' --lda {} --ldb {} --ldc {} --ldd {}'.format(
                    self._args.lda, self._args.ldb, self._args.ldc, self._args.ldd
                )
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
        precision = self._args.precision[cmd_idx]
        self._result.add_raw_data('raw_output_' + precision, raw_output)

        content = raw_output.splitlines()
        gflops_index = None
        gflops = -1

        for line in content:
            try:
                if 'rocblas-Gflops' in line:
                    line = line.split(',')
                    gflops_index = line.index('rocblas-Gflops')
                if gflops_index is not None:
                    line = line.split(',')
                    gflops = float(line[gflops_index])
                if gflops != -1:
                    break
            except BaseException:
                pass

        if gflops == -1:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, raw_output
                )
            )
            return False

        self._result.add_result(precision, gflops)

        return True


BenchmarkRegistry.register_benchmark('gemm-flops', RocmGemmFlops, platform=Platform.ROCM)
