# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU Copy Bandwidth Performance benchmark."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuCopyBwBenchmark(MicroBenchmarkWithInvoke):
    """The GPU copy bandwidth performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'gpu_copy'
        self._mem_types = ['htod', 'dtoh', 'dtod', 'one_to_all', 'all_to_one', 'all_to_all']
        self._copy_types = ['sm', 'dma']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--mem_type',
            type=str,
            nargs='+',
            default=self._mem_types,
            help='Memory types for benchmark. E.g. {}.'.format(' '.join(self._mem_types)),
        )

        self._parser.add_argument(
            '--copy_type',
            type=str,
            nargs='+',
            default=self._copy_types,
            help='Copy types for benchmark. E.g. {}.'.format(' '.join(self._copy_types)),
        )

        self._parser.add_argument(
            '--size',
            type=int,
            default=256 * 1024**2,
            required=False,
            help='Size of data buffer in bytes.',
        )

        self._parser.add_argument(
            '--num_warm_up',
            type=int,
            default=20,
            required=False,
            help='Number of warm up rounds',
        )

        self._parser.add_argument(
            '--num_loops',
            type=int,
            default=100,
            required=False,
            help='Number of data buffer copies performed.',
        )

        self._parser.add_argument(
            '--all_to_all_num_thread_blocks_per_rank',
            type=int,
            default=0,
            required=False,
            help='Number of thread blocks per rank in one-to-all/all-to-one/all-to-all tests.',
        )

        self._parser.add_argument(
            '--all_to_all_thread_block_size',
            type=int,
            default=0,
            required=False,
            help='Thread block size in one-to-all/all-to-one/all-to-all tests.',
        )

        self._parser.add_argument(
            '--bidirectional',
            action='store_true',
            help='Enable bidirectional test',
        )

        self._parser.add_argument(
            '--check_data',
            action='store_true',
            help='Enable data checking',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        args = '--size %d --num_warm_up %d --num_loops %d' % (
            self._args.size, self._args.num_warm_up, self._args.num_loops
        )

        if self._args.all_to_all_num_thread_blocks_per_rank > 0:
            args += ' --all_to_all_num_thread_blocks_per_rank %d' % self._args.all_to_all_num_thread_blocks_per_rank

        if self._args.all_to_all_thread_block_size > 0:
            args += ' --all_to_all_thread_block_size %d' % self._args.all_to_all_thread_block_size

        for mem_type in self._args.mem_type:
            args += ' --%s' % mem_type
        for copy_type in self._args.copy_type:
            args += ' --%s_copy' % copy_type

        if self._args.bidirectional:
            args += ' --bidirectional'

        if self._args.check_data:
            args += ' --check_data'

        self._commands = ['%s %s' % (self.__bin_path, args)]

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
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        try:
            output_lines = [x.strip() for x in raw_output.strip().splitlines()]
            for output_line in output_lines:
                tag, bw_str = output_line.split()
                self._result.add_result(tag + '_bw', float(bw_str))
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('gpu-copy-bw', GpuCopyBwBenchmark)
