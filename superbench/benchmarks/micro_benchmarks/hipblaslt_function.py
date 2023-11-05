# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the hipBlasLt GEMM benchmark."""

import os
import re
import itertools

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class HipBlasLtBenchmark(MicroBenchmarkWithInvoke):
    """The hipBlasLt GEMM benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'hipblaslt-bench'
        self._in_types = ['fp32', 'fp16', 'bf16', 'fp8e4m3']
        self._in_type_map = {
            'fp16': '--a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --compute_type f32_r',
            'fp32': '--a_type f32_r --b_type f32_r --c_type f32_r --d_type f32_r --compute_type f32_r',
            'bf16': '--a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r',
            'fp8e4m3': '--a_type f8_r --b_type f16_r --c_type f8_r --d_type f8_r --compute_type f32_f16_r --scale_type f32_r --bias_vector --bias_type f32_r --api_method cpp'
        }

    def mrange(self, start, stop=-1, multiplication_factor=2, symbol='x'):
        """Range constructor with multiplication factor.

        Args:
            start (int): Start number.
            stop (int, optional): Stop number. Defaults to -1.
            multiplication_factor (int, optional): Multiplication factor. Defaults to 2.

        Yields:
            int: number in the range.
        """
        if symbol == 'x':
            while True:
                yield start
                start *= multiplication_factor
                if start > stop or start == 0 or multiplication_factor < 2:
                    break
        elif symbol == '+':
            while True:
                yield start
                start = start + multiplication_factor
                if start > stop or start == 0 or multiplication_factor < 1:
                    break
        else:
            raise ValueError(f'Invalid symbol {symbol}.')

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
        if len(nums) < 3:
            return all(x.isdigit() for x in nums)
        return nums[0].isdigit() and nums[1].isdigit() and nums[2].lstrip('+').lstrip('x').isdigit()

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
        self._precision_in_commands = []
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
                        *map(lambda shape: self.mrange(
                           *map(lambda dim: int(dim.lstrip('+').lstrip('x')), shape.split(':')) , symbol=shape.split(':')[2][0] if len(shape.split(':')) == 3 and shape.split(':')[2] in ['+', 'x'] else 'x'), shape_list)
                    ):
                        command = f'{self.__bin_path} -m {_m} -n {_n} -k {_k} -j {self._args.num_warmup} -i {self._args.num_steps} {self._in_type_map[_in_type]}'
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

            # Find the line containing 'hipblaslt-Gflops'
            for i, line in enumerate(lines):
                if 'hipblaslt-Gflops' in line:
                    index = i
                    break

            if index is None:
                raise ValueError('Line with "hipblaslt-Gflops" not found in the log.')

            # Split the line into fields using a comma as the delimiter
            fields = lines[index + 1].strip().split(',')

            # Check the number of fields and the format of the first two fields
            if len(fields) != 23 or not all(re.match(r'\d*\.\d*$', item.strip()) or item.strip().isdigit() for item in fields[-2:]):
                raise ValueError('Invalid result')
            
            self._result.add_result(
                f'{self._precision_in_commands[cmd_idx]}_{fields[3]}_{"_".join(fields[4:7])}_flops', float(fields[-2])
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
