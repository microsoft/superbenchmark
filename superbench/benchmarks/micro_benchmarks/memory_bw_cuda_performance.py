# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Cuda memory performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class MemBwCuda(MicroBenchmarkWithInvoke):
    """The Cuda memory bus bandwidth performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'bandwidthTest'
        self.__mem_types = ['--htod', '--dtoh', '--dtod']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        for case in self.__mem_types:
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' ' + case
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

        mb_type = -1
        mem_bw = -1
        metric = ''
        valid = True
        content = raw_output.splitlines()
        try:
            for index, line in enumerate(content):
                if 'Host to Device Bandwidth' in line:
                    mb_type = 1
                elif 'Device to Host Bandwidth' in line:
                    mb_type = 2
                elif 'Device to Device Bandwidth' in line:
                    mb_type = 0
                elif '32000000' in line and mb_type != -1:
                    values = list(filter(None, line.split()))
                    if mb_type == 1:
                        metric = 'H2D_Mem_BW'
                        mem_bw = float(values[1])
                    elif mb_type == 2:
                        metric = 'D2H_Mem_BW'
                        mem_bw = float(values[1])
                    elif mb_type == 0:
                        metric = 'D2D_Mem_BW'
                        mem_bw = float(values[1])

        except BaseException:
            valid = False
        finally:
            if valid is False or mem_bw == -1:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False

        self._result.add_result(metric, mem_bw)

        return True


BenchmarkRegistry.register_benchmark('mem-bw', MemBwCuda, platform=Platform.CUDA)
