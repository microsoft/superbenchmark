# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU SM Copy Bandwidth Performance benchmark."""

import os

from superbench.common.utils import logger
from superbench.common.utils import nv_helper
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class GpuSmCopyBwBenchmark(MicroBenchmarkWithInvoke):
    """The GPU SM copy bandwidth performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'gpu_sm_copy'
        self._mem_types = ['htoh', 'htod', 'dtoh', 'dtod', 'ptop']
        self._num_devices = nv_helper.get_device_count()
        self._command_settings = []

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
            '--size',
            type=int,
            default=64 * 1024**2,
            required=False,
            help='Size of data buffer in bytes.',
        )

        self._parser.add_argument(
            '--num_loops',
            type=int,
            default=100,
            required=False,
            help='Number of data buffer copies performed.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        for mem_type in self._args.mem_type:
            if mem_type == 'htoh':
                self._command_settings += [
                    {'tag': 'htoh_gpu%d' % x, 'devices': ('cpu', 'cpu', 'gpu%d' % x)}
                    for x in range(self._num_devices)]
            elif mem_type == 'htod':
                self._command_settings += [
                    {'tag': 'htod_gpu%d' % x, 'devices': ('cpu', 'gpu%d' % x, 'gpu%d' % x)}
                    for x in range(self._num_devices)]
            elif mem_type == 'dtoh':
                self._command_settings += [
                    {'tag': 'dtoh_gpu%d' % x, 'devices': ('gpu%d' % x, 'cpu', 'gpu%d' % x)}
                    for x in range(self._num_devices)]
            elif mem_type == 'dtod':
                self._command_settings += [
                    {'tag': 'dtod_gpu%d' % x, 'devices': ('gpu%d' % x, 'gpu%d' % x, 'gpu%d' % x)}
                    for x in range(self._num_devices)]
            elif mem_type == 'ptop':
                for x in range(self._num_devices):
                    for y in range(self._num_devices):
                        self._command_settings.append(
                            {'tag': 'ptop_gpu%d_reads_gpu%d' % (y, x), 'devices': ('gpu%d' % x, 'gpu%d' % y, 'gpu%d' % y)})
                        self._command_settings.append(
                            {'tag': 'ptop_gpu%d_writes_gpu%d' % (x, y), 'devices': ('gpu%d' % x, 'gpu%d' % y, 'gpu%d' % x)})
            else:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported mem_type - benchmark: {}, mem_type: {}, expected: {}.'.format(
                        self._name, mem_type, ' '.join(self._mem_types)
                    )
                )
                return False

        self._commands = [
            '%s %s %d %d' % (self.__bin_path, ' '.join(self._command_settings['devices']), self._args.size, self._args.num_loops)
            for x in self._command_settings]

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
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output)

        try:
            output_prefix = 'Bandwidth (GB/s): '
            assert (raw_output.startswith(output_prefix))
            self._result.add_result(self._command_settings[cmd_idx]['tag'], float(raw_output[len(output_prefix):]))
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('gpu-sm-copy-bw', GpuSmCopyBwBenchmark)
