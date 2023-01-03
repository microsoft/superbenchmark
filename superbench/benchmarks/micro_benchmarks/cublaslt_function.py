# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cuBLASLt GEMM benchmark."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CublasLtBenchmark(MicroBenchmarkWithInvoke):
    """The cuBLASLt GEMM benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'cublaslt_fp8_gemm'
        self._in_types = ['fp16', 'fp8e4m3', 'fp8e5m2']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--shapes',
            type=str,
            nargs='+',
            default=[f'{x},{x},{x}' for x in [2048, 4096, 8192]],
            help='Shapes in m,n,k format.',
        )
        self._parser.add_argument(
            '--batch',
            type=int,
            default=0,
            required=False,
            help='Batch size for strided batch GEMM, set 0 to disable.',
        )
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=20,
            required=False,
            help='Number of warm up steps.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=50,
            required=False,
            help='Number of steps to measure.',
        )
        self._parser.add_argument(
            '--in_type',
            type=str,
            default='fp8e4m3',
            required=False,
            help='Input data type, supports {}.'.format(' '.join(self._in_types)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        if self._args.in_type not in self._in_types:
            logger.error(f'Invalid input type {self._args.in_type}.')
            return False

        self._commands = []
        for shape in self._args.shapes:
            shape_list = shape.replace(',', ' ').split()
            if len(shape_list) != 3 or not all(x.isdigit() for x in shape_list):
                logger.error(f'Invalid shape {shape}.')
                return False
            self._commands.append(
                f'{self.__bin_path} -m {shape_list[0]} -n {shape_list[1]} -k {shape_list[2]} '
                f'-b {self._args.batch} -w {self._args.num_warmup} -i {self._args.num_steps} -t {self._args.in_type}'
            )

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
            fields = raw_output.strip().split()
            if len(fields) != 6 or not all(x.isdigit() for x in fields[:4]):
                raise ValueError('Invalid result.')
            self._result.add_result(f'{self._args.in_type}_{"_".join(fields[:3])}_flops', float(fields[-1]))
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('cublaslt-gemm', CublasLtBenchmark, platform=Platform.CUDA)
