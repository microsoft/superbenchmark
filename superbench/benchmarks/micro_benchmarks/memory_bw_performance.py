# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Cuda memory performance benchmarks."""

import os
import re

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
        self.__mem_types = ['htod', 'dtoh', 'dtod']
        self.__memory = ['pageable', 'pinned']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--mem_type',
            type=str,
            nargs='+',
            default=self.__mem_types,
            help='Memory types to benchmark. E.g. {}.'.format(' '.join(self.__mem_types)),
        )
        self._parser.add_argument(
            '--shmoo_mode',
            action='store_true',
            default=False,
            help='Enable shmoo mode for bandwidthtest.',
        )
        self._parser.add_argument(
            '--memory',
            type=str,
            default=None,
            help='Memory argument for bandwidthtest. E.g. {}.'.format(' '.join(self.__memory)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Format the arguments
        if not isinstance(self._args.mem_type, list):
            self._args.mem_type = [self._args.mem_type]
        self._args.mem_type = [p.lower() for p in self._args.mem_type]

        # Check the arguments and generate the commands
        for mem_type in self._args.mem_type:
            if mem_type not in self.__mem_types:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported mem_type of bandwidth test - benchmark: {}, mem_type: {}, expected: {}.'.format(
                        self._name, mem_type, ' '.join(self.__mem_types)
                    )
                )
                return False
            else:
                command = os.path.join(self._args.bin_dir, self._bin_name)
                command += ' --' + mem_type
                if self._args.shmoo_mode:
                    command += ' mode=shmoo'
                if self._args.memory:
                    if self._args.memory in self.__memory:
                        command += ' memory=' + self._args.memory
                    else:
                        self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                        logger.error(
                            'Unsupported memory argument of bandwidth test - benchmark: {}, memory: {}, expected: {}.'.
                            format(self._name, self._args.memory, ' '.join(self.__memory))
                        )
                        return False
                command += ' --csv'
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
        self._result.add_raw_data('raw_output_' + self._args.mem_type[cmd_idx], raw_output)

        mem_bw = -1
        metric = ''
        valid = True
        content = raw_output.splitlines()
        try:
            for index, line in enumerate(content):
                if 'H2D' in line:
                    metric = 'H2D_Mem_BW'
                elif 'D2H' in line:
                    metric = 'D2H_Mem_BW'
                elif 'D2D' in line:
                    metric = 'D2D_Mem_BW'
                else:
                    continue
                line = line.split(',')[1]
                value = re.search(r'(\d+.\d+)', line)
                if value:
                    mem_bw = max(mem_bw, float(value.group(0)))

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
