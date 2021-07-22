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
            help='Nccl operations to benchmark. E.g. {}.'.format(' '.join(list(self.__operations.keys()))),
        )
        self._parser.add_argument(
            '-g',
            type=int,
            default=8,
            help='Number of gpus per thread to run the nccl test.',
        )
        self._parser.add_argument(
            '-e',
            type=str,
            default='8192M',
            help='Max size in bytes to run the nccl test. E.g. 8192M.',
        )
        self._parser.add_argument(
            '-b',
            type=str,
            default='1',
            help='Min size in bytes to run the nccl test. E.g. 1.',
        )
        self._parser.add_argument(
            '-f',
            type=int,
            default=2,
            help='Increment factor, multiplication factor between sizes. E.g. 2.',
        )
        self._parser.add_argument(
            '-c',
            type=int,
            default=0,
            help='Check correctness of results. This can be quite slow on large numbers of GPUs. E.g. 0 or 1.',
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
                command += ' -b {} -e {} -f {} -g {} -c {}'.format(
                    self._args.b, self._args.e, str(self._args.f), str(self._args.g), str(self._args.c)
                )
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
                    # Get first index of condition or default value in list
                    size_index = next((i for i, x in enumerate(line) if x == 'size'), -1)
                    time_index = next((i for i, x in enumerate(line) if x == 'time'), -1)
                    busbw_index = next((i for i, x in enumerate(line) if x == 'busbw'), -1)
                    algbw_index = next((i for i, x in enumerate(line) if x == 'algbw'), -1)
                    break
            if size_index != -1 and busbw_index != -1 and time_index != -1 and algbw_index != -1:
                for line in content:
                    line = line.strip(' ')
                    line = re.sub(r' +', ' ', line).split(' ')
                    # Filter line not started with number
                    if not re.match(r'\d+', line[0]):
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
