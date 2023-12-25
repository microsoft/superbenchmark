# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the hipBlasLt GEMM benchmark."""

import os

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
            if len(fields) != 23:
                raise ValueError('Invalid result')

            self._result.add_result(
                f'{self._precision_in_commands[cmd_idx]}_{fields[3]}_{"_".join(fields[4:7])}_flops',
                float(fields[-2]) / 1000
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
