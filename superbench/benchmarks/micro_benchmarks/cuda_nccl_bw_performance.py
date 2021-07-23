# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the NCCL performance benchmarks."""

import os
import re

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CudaNcclBwBenchmark(MicroBenchmarkWithInvoke):
    """The NCCL bus bandwidth performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'all_reduce_perf'
        self.__operations = {
            'allreduce': 'all_reduce_perf',
            'allgather': 'all_gather_perf',
            'broadcast': 'broadcast_perf',
            'reduce': 'reduce_perf',
            'reducescatter': 'reduce_scatter_perf',
            'alltoall': 'alltoall_perf'
        }

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--operations',
            type=str,
            nargs='+',
            default=list(self.__operations.keys()),
            help='Nccl operations to benchmark, e.g., {}.'.format(' '.join(list(self.__operations.keys()))),
        )
        self._parser.add_argument(
            '--nccl_tests_args',
            type=str,
            default='-b 8 -e 8G -f 2 -g 8 -c 0',
            help='The arguments for nccl-tests, e.g., -b 8 -e 8G -f 2 -g 8 -c 0.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Format the arguments
        self._args.operations = [p.lower() for p in self._args.operations]

        # Check the arguments and generate the commands
        for op in self._args.operations:
            if op not in self.__operations:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported operation of NCCL test - benchmark: {}, operation: {}, expected: {}.'.format(
                        self._name, op, ' '.join(list(self.__operations.keys()))
                    )
                )
                return False
            else:
                self._bin_name = self.__operations[op]
                if not self._set_binary_path():
                    return False

                command = os.path.join(self._args.bin_dir, self._bin_name)
                command += ' ' + self._args.nccl_tests_args
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
        # If it's invoked by MPI and rank is not 0, empty content is expected
        if os.getenv('OMPI_COMM_WORLD_RANK'):
            rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            if rank > 0:
                return True

        self._result.add_raw_data('raw_output_' + self._args.operations[cmd_idx], raw_output)

        content = raw_output.splitlines()
        size = -1
        busbw_out = -1
        time_out = -1
        algbw_out = -1
        try:
            # Filter useless output
            out_of_place_index = -1
            out_of_bound_index = -1
            for index, line in enumerate(content):
                if 'out-of-place' in line:
                    out_of_place_index = index
                if 'Out of bounds values' in line:
                    out_of_bound_index = index
            content = content[out_of_place_index + 1:out_of_bound_index]
            # Parse max out of bound bus bw as the result
            size_index = -1
            time_index = -1
            busbw_index = -1
            algbw_index = -1
            for line in content:
                if 'time' in line and 'busbw' in line:
                    # Get index of selected column
                    line = line[1:].strip(' ')
                    line = re.sub(r' +', ' ', line).split(' ')
                    # Get first index of condition in list, if it not existing, raise exception
                    size_index = line.index('size') - len(line)
                    time_index = line.index('time') - len(line)
                    busbw_index = line.index('busbw') - len(line)
                    algbw_index = line.index('algbw') - len(line)
                    break
            if size_index != -1 and busbw_index != -1 and time_index != -1 and algbw_index != -1:
                for line in content:
                    line = line.strip(' ')
                    line = re.sub(r' +', ' ', line).split(' ')
                    # Filter line not started with number
                    if len(line) == 0 or not re.match(r'\d+', line[0]):
                        continue
                    size = int(line[size_index])
                    if size != 0:
                        busbw_out = float(line[busbw_index])
                        time_out = float(line[time_index])
                        algbw_out = float(line[algbw_index])
                        self._result.add_result(
                            'NCCL_' + self._args.operations[cmd_idx] + '_' + str(size) + '_busbw', busbw_out
                        )
                        self._result.add_result(
                            'NCCL_' + self._args.operations[cmd_idx] + '_' + str(size) + '_algbw', algbw_out
                        )
                        self._result.add_result(
                            'NCCL_' + self._args.operations[cmd_idx] + '_' + str(size) + '_time', time_out
                        )
        except BaseException as e:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False
        if out_of_place_index == -1 or out_of_bound_index == -1 or busbw_out == -1:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, raw_output
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('nccl-bw', CudaNcclBwBenchmark, platform=Platform.CUDA)
