# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the NCCL performance benchmarks."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class NcclBw(MicroBenchmarkWithInvoke):
    """The NCCL bus bandwidth performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'all_reduce_perf'
        self.__bin_list = ['all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--bin_list',
            type=str,
            nargs='+',
            default=self.__bin_list,
            help='Precision for benchmarking. E.g. {}.'.format(' '.join(self.__bin_list)),
        )

        self._parser.add_argument(
            '--gpu_count',
            type=int,
            default=8,
            help='GPU num for benchmarking.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self._args.bin_list = [p.lower() for p in self._args.bin_list]
        for bin_name in self._args.bin_list:
            if bin_name not in self.__bin_list:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported bin of NCCL test - benchmark: {}, bin name: {}, expected: {}.'.format(
                        self._name, bin_name, self.__bin_list)
                )
                return False
            else:
                self._bin_name = bin_name
                if not self._set_binary_path():
                    return False

                command = 'NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 '
                command += os.path.join(self._args.bin_dir, self._bin_name)
                command += '-b 1 -e 8192M -f 2 -g {} -c 0 '.format(self._args.gpu_count)
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
        self._result.add_raw_data('raw_output_' + self._args.bin_list[cmd_idx], raw_output)

        #algbw_out = 0
        busbw_out = 0
        content = raw_output.splitlines()
        try:
            for line in content:          
                line = line.strip(' ')
                line = re.sub(r' +', ' ', line).split(' ')
                if len(line) <= 10:
                    break
                #algbw_out = max(algbw_out, float(line[-7]))
                busbw_out = max(busbw_out, float(line[-6]))
        except BaseException:
            valid = False
        finally:
            if valid is False or busbw_out == 0:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False

        self._result.add_resultself._args.bin_list[cmd_idx], busbw_out)

        return True


BenchmarkRegistry.register_benchmark('nccl-bw', NcclBw, platform=Platform.CUDA)
