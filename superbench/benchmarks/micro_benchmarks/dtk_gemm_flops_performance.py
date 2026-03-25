# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FLOPs performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
from superbench.benchmarks.micro_benchmarks import GemmFlopsBenchmark


class DtkGemmFlopsBenchmark(GemmFlopsBenchmark):
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
            '--iterations',
            type=int,
            default=10,
            required=False,
            help='Iterations to run inside the timing loop.',
        )
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
            default='N',
            help='Transpose type of Matrix B, N = no transpose, T = transpose, C = conjugate transpose',
        )
        self._parser.add_argument(
            '--side',
            type=str.upper,
            choices=['L', 'R'],
            default='L',
            required=False,
            help='Side selection for routines that support it.',
        )
        self._parser.add_argument(
            '--uplo',
            type=str.upper,
            choices=['U', 'L'],
            default='U',
            required=False,
            help='Triangle selection for routines that support it.',
        )
        self._parser.add_argument(
            '--diag',
            type=str.upper,
            choices=['U', 'N'],
            default='N',
            required=False,
            help='Unit or non-unit diagonal for routines that support it.',
        )
        self._parser.add_argument(
            '--lda',
            type=int,
            default=16384,
            required=False,
            help='Leading dimension of matrix A.',
        )
        self._parser.add_argument(
            '--ldb',
            type=int,
            default=16384,
            required=False,
            help='Leading dimension of matrix B.',
        )
        self._parser.add_argument(
            '--ldc',
            type=int,
            default=16384,
            required=False,
            help='Leading dimension of matrix C.',
        )
        self._parser.add_argument(
            '--ldd',
            type=int,
            default=16384,
            required=False,
            help='Leading dimension of matrix D.',
        )
        self._parser.add_argument(
            '--any_stride',
            action='store_true',
            help='Do not modify input strides based on leading dimensions.',
        )
        self._parser.add_argument(
            '--stride_a',
            type=int,
            required=False,
            help='Specific stride of strided_batched matrix A.',
        )
        self._parser.add_argument(
            '--stride_b',
            type=int,
            required=False,
            help='Specific stride of strided_batched matrix B.',
        )
        self._parser.add_argument(
            '--stride_c',
            type=int,
            required=False,
            help='Specific stride of strided_batched matrix C.',
        )
        self._parser.add_argument(
            '--stride_d',
            type=int,
            required=False,
            help='Specific stride of strided_batched matrix D.',
        )
        self._parser.add_argument(
            '--kl',
            type=int,
            default=32,
            required=False,
            help='Number of sub-diagonals for routines that support banded matrices.',
        )
        self._parser.add_argument(
            '--ku',
            type=int,
            default=32,
            required=False,
            help='Number of super-diagonals for routines that support banded matrices.',
        )
        self._parser.add_argument(
            '--alpha',
            type=float,
            default=1.0,
            required=False,
            help='Specifies the scalar alpha.',
        )
        self._parser.add_argument(
            '--beta',
            type=float,
            default=0.0,
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
        self._parser.add_argument(
            '--verify',
            type=int,
            default=0,
            choices=[0, 1],
            required=False,
            help='Validate GPU results with CPU. 0 = No, 1 = Yes.',
        )
        self._parser.add_argument(
            '--outofplace',
            action='store_true',
            help='Use separate input/output buffers for gemm_ex out-of-place execution.',
        )
        self._parser.add_argument(
            '--algo',
            type=int,
            default=0,
            required=False,
            help='Extended precision gemm algorithm.',
        )
        self._parser.add_argument(
            '--solution_index',
            type=int,
            default=0,
            required=False,
            help='Extended precision gemm solution index.',
        )
        self._parser.add_argument(
            '--flags',
            type=int,
            default=0,
            required=False,
            help='gemm_ex flags.',
        )
        self._parser.add_argument(
            '--workspace',
            type=int,
            default=0,
            required=False,
            help='Fixed workspace memory size.',
        )
        self._parser.add_argument(
            '--math_mode',
            type=int,
            default=0,
            required=False,
            help='Extended precision gemm math mode.',
        )
        self._parser.add_argument(
            '--flush_batch_count',
            type=int,
            default=1,
            required=False,
            help='Number of copies of arrays to allocate for cache flushing in timing code.',
        )
        self._parser.add_argument(
            '--flush_memory_size',
            type=int,
            default=0,
            required=False,
            help='Bytes of memory occupied by arrays for cache flushing in timing code.',
        )
        self._parser.add_argument(
            '--atomics_allowed',
            action='store_true',
            help='Allow atomic operations with non-determinism in results.',
        )
        self._parser.add_argument(
            '--atomics_not_allowed',
            action='store_true',
            help='Disallow atomic operations with non-determinism in results.',
        )
        self._parser.add_argument(
            '--device',
            type=int,
            default=0,
            required=False,
            help='Set default device to be used for the benchmark process.',
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
            command += ' --iters {}'.format(self._args.iterations)
            command += ' --cold_iters {}'.format(self._args.num_warmup)
            command += ' --transposeA {} --transposeB {}'.format(self._args.transposeA, self._args.transposeB)
            command += ' --side {} --uplo {} --diag {}'.format(self._args.side, self._args.uplo, self._args.diag)
            command += ' -m {} -n {} -k {}'.format(self._args.m, self._args.n, self._args.k)
            command += ' --alpha {} --beta {}'.format(self._args.alpha, self._args.beta)
            command += ' --kl {} --ku {}'.format(self._args.kl, self._args.ku)
            command += ' --lda {} --ldb {} --ldc {} --ldd {}'.format(
                self._args.lda, self._args.ldb, self._args.ldc, self._args.ldd
            )
            if self._args.any_stride:
                command += ' --any_stride'
            if self._args.stride_a is not None:
                command += ' --stride_a {}'.format(self._args.stride_a)
            if self._args.stride_b is not None:
                command += ' --stride_b {}'.format(self._args.stride_b)
            if self._args.stride_c is not None:
                command += ' --stride_c {}'.format(self._args.stride_c)
            if self._args.stride_d is not None:
                command += ' --stride_d {}'.format(self._args.stride_d)
            command += ' --verify {}'.format(self._args.verify)
            if self._args.outofplace:
                command += ' --outofplace'
            command += ' --algo {}'.format(self._args.algo)
            command += ' --solution_index {}'.format(self._args.solution_index)
            command += ' --flags {}'.format(self._args.flags)
            command += ' --workspace {}'.format(self._args.workspace)
            command += ' --math_mode {}'.format(self._args.math_mode)
            command += ' --flush_batch_count {}'.format(self._args.flush_batch_count)
            command += ' --flush_memory_size {}'.format(self._args.flush_memory_size)
            if self._args.atomics_allowed:
                command += ' --atomics_allowed'
            if self._args.atomics_not_allowed:
                command += ' --atomics_not_allowed'
            command += ' --device {}'.format(self._args.device)
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


BenchmarkRegistry.register_benchmark('gemm-flops', DtkGemmFlopsBenchmark, platform=Platform.DTK)
