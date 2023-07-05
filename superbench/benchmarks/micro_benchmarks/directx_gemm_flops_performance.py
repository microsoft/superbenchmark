# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the DirectXGPUCoreFlops performance benchmarks."""

import os
from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class DirectXGPUCoreFlops(MicroBenchmarkWithInvoke):
    """The DirectXGPUCoreFlops benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)
        self._bin_name = 'DirectXGPUCoreFlops.exe'
        self._support_precisions = ['fp16', 'fp32']
        self._precision_need_to_run = list()

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--num_loops',
            type=int,
            default=10,
            required=False,
            help='The number of benchmark runs.',
        )
        self._parser.add_argument(
            '--num_warm_up',
            type=int,
            default=2,
            required=False,
            help='The number of warm up runs.',
        )
        self._parser.add_argument(
            '--n',
            type=int,
            default=16 * 256,
            required=False,
            help='The N dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--k',
            type=int,
            default=16 * 256,
            required=False,
            help='The K dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--m',
            type=int,
            default=16 * 256,
            required=False,
            help='The M dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--precision',
            type=str,
            nargs='+',
            default=list(),
            help='Precision for benchmarking. E.g. {}.'.format(' '.join(self._support_precisions)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if len(self._args.precision) == 0:
            self._precision_need_to_run = self._support_precisions
        else:
            self._args.precision = [p.lower() for p in self._args.precision]
            for p in self._args.precision:
                if p not in self._support_precisions:
                    logger.warning(
                        'Unsupported precision - benchmark: {}, precision: {}, expected: {}.'.format(
                            self._name, p, self._support_precisions
                        )
                    )
                else:
                    self._precision_need_to_run.append(p)

        if len(self._precision_need_to_run) == 0:
            self._result.set_return_code(ReturnCode.NO_SUPPORTED_PRECISION)
            return False

        for p in self._precision_need_to_run:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += (' --num_loops ' + str(self._args.num_loops))
            command += (' --num_warm_up ' + str(self._args.num_warm_up))
            command += (' --n ' + str(self._args.n))
            command += (' --k ' + str(self._args.k))
            command += (' --m ' + str(self._args.m))
            command += (' --' + p)
            self._commands.append(command)
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        precision = self._precision_need_to_run[cmd_idx]
        self._result.add_raw_data('raw_output_' + precision, raw_output, self._args.log_raw_data)
        valid = True
        flops = list()
        content = raw_output.splitlines()
        try:
            for line in content:
                if 'TFLOPs' in line:
                    flops.append(float(line.split()[0]))
        except BaseException:
            valid = False
        finally:
            if valid is False or len(flops) == 0:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False
        self._result.add_result(precision + '_flops', max(flops))
        return True


BenchmarkRegistry.register_benchmark('directx-gpu-core-flops', DirectXGPUCoreFlops, platform=Platform.DIRECTX)
