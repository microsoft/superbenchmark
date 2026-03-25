# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FLOPs performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import GemmFlopsBenchmark


class RocmGemmFlopsBenchmark(GemmFlopsBenchmark):
    """The GEMM FLOPs performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'rocblas-bench'
        self._support_precisions = ['fp64', 'fp32_xdlops', 'fp16_xdlops', 'bf16_xdlops', 'int8_xdlops']
        self.__precision_and_kernel_map = {
            'fp64': '-r f64_r -f gemm',
            'fp32_xdlops': '-r f32_r -f gemm_ex --compute_type f32_r',
            'fp16_xdlops': '-r f16_r -f gemm_ex --compute_type f32_r',
            'bf16_xdlops': '-r bf16_r -f gemm_ex --compute_type f32_r',
            'int8_xdlops': '--a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r -f gemm_ex --compute_type i32_r'
        }

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--transposeA',
            type=str.upper,
            choices=['N', 'T', 'C'],
            default='N',
            help='Transpose type of Matrix A, N = no transpose, T = transpose, C = conjugate transpose',
        )
        self._parser.add_argument(
            '--transposeB',
            type=str.upper,
            choices=['N', 'T', 'C'],
            default='T',
            help='Transpose type of Matrix B, N = no transpose, T = transpose, C = conjugate transpose',
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
        self._parser.add_argument(
            '--initialization',
            type=str,
            default='hpl',
            choices=['rand_int', 'trig_float', 'hpl'],
            required=False,
            help='Initialize with random integers, trig functions sin and cos, or hpl-like input.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        for p in self._precision_need_to_run:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' ' + self.__precision_and_kernel_map[p]
            command += ' --transposeA {} --transposeB {}'.format(self._args.transposeA, self._args.transposeB)
            command += ' -m {} -n {} -k {}'.format(self._args.m, self._args.n, self._args.k)
            command += ' --alpha {} --beta {}'.format(self._args.alpha, self._args.beta)
            command += ' --lda {} --ldb {} --ldc {} --ldd {}'.format(
                self._args.lda, self._args.ldb, self._args.ldc, self._args.ldd
            )
            command += ' --initialization {}'.format(self._args.initialization)
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
        precision = self._precision_need_to_run[cmd_idx]
        self._result.add_raw_data('raw_output_' + precision, raw_output, self._args.log_raw_data)

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

        self._result.add_result(self._metric_map[precision], gflops)

        return True


BenchmarkRegistry.register_benchmark('gemm-flops', RocmGemmFlopsBenchmark, platform=Platform.ROCM)
