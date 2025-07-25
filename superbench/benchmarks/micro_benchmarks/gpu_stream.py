# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU Stream Performance benchmark."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuStreamBenchmark(MicroBenchmarkWithInvoke):
    """The GPU stream performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'gpu_stream'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--size',
            type=int,
            default=4096 * 1024**2,
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

        args = '--size %d --num_warm_up %d --num_loops %d ' % (
            self._args.size, self._args.num_warm_up, self._args.num_loops
        )

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
            count = 0
            for output_line in output_lines:
                if output_line.startswith('STREAM_'):
                    count += 1
                    tag, bw_str, ratio = output_line.split()
                    self._result.add_result(tag + '_bw', float(bw_str))
                    self._result.add_result(tag + '_ratio', float(ratio))
            if count == 0:
                raise BaseException('No valid results found.')
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('gpu-stream', GpuStreamBenchmark)
