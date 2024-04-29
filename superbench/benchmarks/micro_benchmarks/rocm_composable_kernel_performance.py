# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the ROCm composable kernel GEMM benchmark."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import BlasLtBaseBenchmark


class RocmComposableKernelBenchmark(BlasLtBaseBenchmark):
    """The composable kernel GEMM benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'ckProfiler'
        self._in_types = ['fp32', 'fp16', 'bf16', 'fp8', 'int8']
        self._in_type_map = {
            'fp16': '1',
            'fp32': '0',
            'bf16': '2',
            'fp8': '4',
            'int8': '3',
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
            default='int',
            choices=['float', 'int'],
            required=False,
            help='Initialize matrix data.',
        )
        self._parser.add_argument(
            '--matrixA_layout',
            type=str,
            default='row',
            choices=['row', 'col'],
            required=False,
            help='Matrix A Layout. RowMajor or ColMajor.',
        )
        self._parser.add_argument(
            '--matrixB_layout',
            type=str,
            default='row',
            choices=['row', 'col'],
            required=False,
            help='Matrix B Layout. RowMajor or ColMajor.',
        )
        self._parser.add_argument(
            '--check_data',
            action='store_true',
            required=False,
            help='Whether check data correctness.',
        )
        self._parser.add_argument(
            '--splitk',
            type=int,
            default=None,
            required=False,
            nargs='+',
            help='Split K dimension.',
        )
        self._parser.add_argument(
            '--streamk',
            type=int,
            default=None,
            required=False,
            nargs='+',
            help='Stream K blocks.',
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
        matrix_layout = '0'
        if self._args.matrixA_layout == 'row' and self._args.matrixB_layout == 'row':
            matrix_layout = '0'
        elif self._args.matrixA_layout == 'row' and self._args.matrixB_layout == 'col':
            matrix_layout = '1'
        elif self._args.matrixA_layout == 'col' and self._args.matrixB_layout == 'row':
            matrix_layout = '2'
        elif self._args.matrixA_layout == 'col' and self._args.matrixB_layout == 'col':
            matrix_layout = '3'
        if self._args.check_data:
            self._args.check_data = '1'
        else:
            self._args.check_data = '0'
        init = 1 if self._args.initialization == 'int' else 2
        for (_m, _n, _k, _b, _in_type) in self._shapes_to_run:
            params = f'{self._in_type_map[_in_type]}' + \
                f' {matrix_layout} {self._args.check_data} {init} 0 1' + \
                f' {_m} {_n} {_k} -1 -1 -1'
            command = f'{self.__bin_path} gemm {params} {self._args.num_warmup} {self._args.num_steps}'
            self._commands.append(command)
            logger.info(command)
            if self._args.splitk and _in_type not in ['fp8']:
                if not isinstance(self._args.splitk, list):
                    self._args.splitk = [self._args.splitk]
                for splitk in self._args.splitk:
                    command = f'{self.__bin_path} gemm_splitk {params} {splitk}' + \
                        f' {self._args.num_warmup} {self._args.num_steps}'
                    self._commands.append(command)
                    logger.info(command)
            if self._args.streamk and _in_type not in ['fp8']:
                if not isinstance(self._args.streamk, list):
                    self._args.streamk = [self._args.streamk]
                for streamk in self._args.streamk:
                    command = f'{self.__bin_path} gemm_streamk {params} {streamk}' + \
                        f' {self._args.num_warmup} {self._args.num_steps}'
                    self._commands.append(command)
                    logger.info(command)
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
                if 'Best Perf' in line:
                    index = i
                    break

            if index is not None:
                # Search the text for each pattern
                datatype_match = re.search(r"datatype = (\w+)", line)
                m_match = re.search(r"M = (\d+)", line)
                n_match = re.search(r"N = (\d+)", line)
                k_match = re.search(r"K = (\d+)", line)
                flops_match = re.search(r"(\d+\.?\d*) TFlops", line)

                # Extract the matched groups
                datatype = datatype_match.group(1) if datatype_match else None
                m = int(m_match.group(1)) if m_match else None
                n = int(n_match.group(1)) if n_match else None
                k = int(k_match.group(1)) if k_match else None
                flops = float(flops_match.group(1)) if flops_match else None

                metric = f'{datatype}_{m}_{n}_{k}_flops'
                self._result.add_result(metric, flops)
            else:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False

        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False
        finally:
            if cmd_idx == len(self._commands) - 1:
                for metric in self._result.result:
                    self._result.result[metric] = [max(self._result.result[metric])]
        return True


BenchmarkRegistry.register_benchmark('composable-kernel-gemm', RocmComposableKernelBenchmark, platform=Platform.ROCM)
