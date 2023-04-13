# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cuBLASLt GEMM benchmark."""

import os
import itertools

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

        self._bin_name = 'cublaslt_gemm'
        self._in_types = ['fp64', 'fp32', 'fp16', 'bf16', 'fp8e4m3', 'fp8e5m2']

    def mrange(self, start, stop=-1, multiplication_factor=2):
        """Range constructor with multiplication factor.

        Args:
            start (int): Start number.
            stop (int, optional): Stop number. Defaults to -1.
            multiplication_factor (int, optional): Multiplication factor. Defaults to 2.

        Yields:
            int: number in the range.
        """
        while True:
            yield start
            start *= multiplication_factor
            if start > stop or start == 0 or multiplication_factor < 2:
                break

    def validate_mrange(self, string):
        """Validate mrange string in format start[[:stop]:multiplication_factor].

        Args:
            string (str): mrange string.

        Returns:
            bool: whether the mrange is expected.
        """
        nums = string.split(':')
        if len(nums) > 3:
            return False
        return bool(all(x.isdigit() for x in nums))

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--shapes',
            type=str,
            nargs='+',
            default=[f'{x},{x},{x}' for x in [2048, 4096, 8192]],
            help='Shapes in m,n,k format. Support format start:stop:multiplication_factor, e.g., 16:128:2.',
        )
        self._parser.add_argument(
            '--batch',
            type=str,
            default='0',
            required=False,
            help=(
                'Batch size for strided batch GEMM, set 0 to disable.'
                ' Support format start:stop:multiplication_factor, e.g., 16:128:2.'
            ),
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
            '--in_types',
            type=str,
            nargs='+',
            default=['fp8e4m3'],
            required=False,
            help='List of input data types, support {}.'.format(' '.join(self._in_types)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        if not self.validate_mrange(self._args.batch):
            logger.error(f'Invalid batch size {self._args.batch}.')
            return False

        self._commands = []
        for _in_type in self._args.in_types:
            if _in_type not in self._in_types:
                logger.error(f'Invalid input type {_in_type}.')
                return False
            for _b in self.mrange(*map(int, self._args.batch.split(':'))):
                for shape in self._args.shapes:
                    shape_list = shape.replace(',', ' ').split()
                    if len(shape_list) != 3 or not all(self.validate_mrange(x) for x in shape_list):
                        logger.error(f'Invalid shape {shape}.')
                        return False
                    for _m, _n, _k in itertools.product(
                        *map(lambda shape: self.mrange(*map(int, shape.split(':'))), shape_list)
                    ):
                        self._commands.append(
                            f'{self.__bin_path} -m {_m} -n {_n} -k {_k} -b {_b} '
                            f'-w {self._args.num_warmup} -i {self._args.num_steps} -t {_in_type}'
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
            self._result.add_result(
                f'{self._commands[cmd_idx].split()[-1]}_{fields[3]}_{"_".join(fields[:3])}_flops', float(fields[-1])
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


BenchmarkRegistry.register_benchmark('cublaslt-gemm', CublasLtBenchmark, platform=Platform.CUDA)
