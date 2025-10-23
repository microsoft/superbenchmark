# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cuBLASLt GEMM benchmark."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import BlasLtBaseBenchmark


class CublasLtBenchmark(BlasLtBaseBenchmark):
    """The cuBLASLt GEMM benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'cublaslt_gemm'
        self._in_types = ['fp64', 'fp32', 'fp16', 'bf16', 'fp8e4m3', 'fp8e5m2', 'fp4e2m1', 'int8']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--in_types',
            type=str,
            nargs='+',
            default=['fp8e4m3'],
            required=False,
            help='List of input data types, support {}.'.format(' '.join(self._in_types)),
        )
        self._parser.add_argument(
            '--enable_autotune',
            action='store_true',
            required=False,
            help='Enable exhaustive autotune mode to find best algorithm.',
        )
        self._parser.add_argument(
            '--num_warmup_autotune',
            type=int,
            default=20,
            required=False,
            help='Number of warm up steps for autotune.',
        )
        self._parser.add_argument(
            '--num_steps_autotune',
            type=int,
            default=50,
            required=False,
            help='Number of steps to measure for autotune.',
        )
        self._parser.add_argument(
            '--enable_ncu_profiling',
            action='store_true',
            required=False,
            help='Enable ncu profiling for each run.',
        )
        self._parser.add_argument(
            '--profiling_metrics',
            type=str,
            nargs='+',
            default=None,
            required=False,
            help='List of ncu profiling metrics, support all ncu metrics.',
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
        for _m, _n, _k, _b, _in_type in self._shapes_to_run:
            # pull out the autotune args onto their own short f-string
            autotune_args = (
                f' -a -W {self._args.num_warmup_autotune}'
                f' -I {self._args.num_steps_autotune}'
            ) if self._args.enable_autotune else ''
            command = f'{self.__bin_path} -m {_m} -n {_n} -k {_k} -b {_b} ' + \
                f'-w {self._args.num_warmup} -i {self._args.num_steps} -t {_in_type}' + \
                f'{(" " + autotune_args) if autotune_args else ""}'
            if self._args.enable_ncu_profiling:
                skip_num = self._args.num_warmup - 1 if self._args.num_warmup > 1 else 0
                command = f'ncu --set full --launch-skip {skip_num} --launch-count 1 --csv ' + command
            self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):    # noqa: C901
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
            if not self._args.enable_ncu_profiling:
                fields = raw_output.strip().split()
                if len(fields) != 6 or not all(x.isdigit() for x in fields[:4]):
                    raise ValueError('Invalid result.')
                self._result.add_result(
                    f'{self._commands[cmd_idx].split()[-1]}_{fields[3]}_{"_".join(fields[:3])}_flops',
                    float(fields[-1])
                )
            else:
                lines = raw_output.strip().split('\n')
                # find line index of the line that starts with "ID","Process ID"
                start_idx = next(i for i, line in enumerate(lines) if 'Metric Name' in line)
                if start_idx == 0 or start_idx == len(lines) - 1:
                    raise ValueError('Invalid result.')
                result_lines = lines[0:start_idx - 1]
                result = False
                size = ''
                for line in result_lines:
                    fields = line.strip().split()
                    if len(fields) == 6 and all(x.isdigit() for x in fields[:4]):
                        result = True
                        size = f'{fields[3]}_{"_".join(fields[:3])}'
                        self._result.add_result(
                            f'{self._commands[cmd_idx].split()[-1]}_{fields[3]}_{"_".join(fields[:3])}_flops',
                            float(fields[-1])
                        )
                if not result:
                    raise ValueError('Invalid result.')
                metric_name_index = lines[start_idx].strip().split(',').index('"Metric Name"')
                metric_value_index = lines[start_idx].strip().split(',').index('"Metric Value"')
                if metric_name_index < 0 or metric_value_index < 0:
                    raise ValueError('Can not find Metric Name and Value.')
                for line in lines[start_idx + 1:]:
                    fields = line.strip().split('","')
                    metric_name = fields[metric_name_index].strip('"').replace(' ', '_')
                    if len(fields) < 15:
                        continue
                    if not self._args.profiling_metrics or metric_name in self._args.profiling_metrics:
                        value = fields[metric_value_index].strip(',').strip('"')
                        try:
                            float_value = float(value)
                            self._result.add_result(
                                f'{self._commands[cmd_idx].split()[-1]}_{size}_{metric_name}', float_value
                            )
                        except ValueError:
                            pass
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
