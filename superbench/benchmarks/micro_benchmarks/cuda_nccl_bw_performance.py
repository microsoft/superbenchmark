# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the NCCL/RCCL performance benchmarks.

We assume NCCL-tests and RCCL-tests have the same interface and output in the test scope so far.
So the arguments and result parsing are the same.
"""

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
            '--operation',
            type=str,
            default='allreduce',
            help='NCCL operation to benchmark, e.g., {}.'.format(' '.join(list(self.__operations.keys()))),
        )
        self._parser.add_argument(
            '--ngpus',
            type=int,
            default=1,
            help='Number of gpus per thread to run the nccl test.',
        )
        self._parser.add_argument(
            '--maxbytes',
            type=str,
            default='8G',
            help='Max size in bytes to run the nccl test. E.g. 8G.',
        )
        self._parser.add_argument(
            '--minbytes',
            type=str,
            default='8',
            help='Min size in bytes to run the nccl test. E.g. 1.',
        )
        self._parser.add_argument(
            '--stepfactor',
            type=int,
            default=2,
            help='Increment factor, multiplication factor between sizes. E.g. 2.',
        )
        self._parser.add_argument(
            '--check',
            type=int,
            default=0,
            help='Check correctness of results. This can be quite slow on large numbers of GPUs. E.g. 0 or 1.',
        )
        self._parser.add_argument(
            '--iters',
            type=int,
            default=20,
            help='Number of iterations. Default: 20.',
        )
        self._parser.add_argument(
            '--warmup_iters',
            type=int,
            default=5,
            help='Number of warmup iterations. Default: 5.',
        )
        self._parser.add_argument(
            '--graph_iters',
            type=int,
            default=0,
            help='Number of graph launch iterations. Set to 0 to disable graph mode. Default: 0.',
        )
        self._parser.add_argument(
            '--in_place',
            action='store_true',
            help='If specified, collect in-place numbers, else collect out-of-place numbers.',
        )
        self._parser.add_argument(
            '--data_type',
            type=str,
            default='float',
            help='Data type used in NCCL operations. Default: float.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Format the arguments
        self._args.operation = self._args.operation.lower()

        # Check the arguments and generate the commands
        op = self._args.operation
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
            command += ' -b {} -e {} -f {} -g {} -c {} -n {} -w {} -G {} -d {}'.format(
                self._args.minbytes, self._args.maxbytes, str(self._args.stepfactor), str(self._args.ngpus),
                str(self._args.check), str(self._args.iters), str(self._args.warmup_iters), str(self._args.graph_iters),
                self._args.data_type
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

        self._result.add_raw_data('raw_output_' + self._args.operation, raw_output, self._args.log_raw_data)

        content = raw_output.splitlines()
        size = -1
        busbw_out = -1
        time_out = -1
        algbw_out = -1
        serial_index = os.environ.get('SB_MODE_SERIAL_INDEX', -1)
        parallel_index = os.environ.get('SB_MODE_PARALLEL_INDEX', -1)

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
            time_index = None
            busbw_index = None
            algbw_index = None
            for line in content:
                if 'time' in line and 'busbw' in line:
                    # Get index of selected column
                    line = line[1:].strip(' ')
                    line = re.sub(r' +', ' ', line).split(' ')
                    # Get first index of condition in list, if it not existing, raise exception
                    size_index = line.index('size')
                    # Need index from the end because sometimes previous fields (like redop) can be empty
                    if self._args.in_place:
                        time_index = -1 - list(reversed(line)).index('time')
                        busbw_index = -1 - list(reversed(line)).index('busbw')
                        algbw_index = -1 - list(reversed(line)).index('algbw')
                    else:
                        time_index = line.index('time') - len(line)
                        busbw_index = line.index('busbw') - len(line)
                        algbw_index = line.index('algbw') - len(line)
                    break
            if size_index != -1 and busbw_index is not None and time_index is not None and algbw_index is not None:
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
                        exec_index = '_{}_{}:'.format(
                            serial_index, parallel_index
                        ) if serial_index != -1 and parallel_index != -1 else '_'
                        prefix_name = '{}{}{}_'.format(self._args.operation, exec_index, size)
                        self._result.add_result(prefix_name + 'busbw', busbw_out)
                        self._result.add_result(prefix_name + 'algbw', algbw_out)
                        self._result.add_result(prefix_name + 'time', time_out)
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
BenchmarkRegistry.register_benchmark('rccl-bw', CudaNcclBwBenchmark, platform=Platform.ROCM)
