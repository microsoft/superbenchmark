# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU SM Copy Bandwidth Performance benchmark."""

import os

from superbench.common.utils import logger
from superbench.common.devices import GPU
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
        self._num_devices = GPU().count
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

    def _gen_command_setting(self, tag, src_dev, dst_dev, working_dev):
        """Generate command setting structure.

        Return:
            Command settings.
        """
        return {'tag': tag, 'devices': (src_dev, dst_dev, working_dev)}

    def _gen_htoh_command_settings(self):
        """Generate host-to-host command settings.

        Return:
            Host-to-host command settings.
        """
        return [
            self._gen_command_setting('htoh_gpu%d' % x, 'cpu', 'cpu', 'gpu%d' % x)
            for x in range(self._num_devices)
        ]

    def _gen_htod_command_settings(self):
        """Generate host-to-device command settings.

        Return:
            Host-to-device command settings.
        """
        return [
            self._gen_command_setting('htod_gpu%d' % x, 'cpu', 'gpu%d' % x, 'gpu%d' % x)
            for x in range(self._num_devices)
        ]

    def _gen_dtoh_command_settings(self):
        """Generate device-to-host command settings.

        Return:
            Device-to-host command settings.
        """
        return [
            self._gen_command_setting('dtoh_gpu%d' % x, 'gpu%d' % x, 'cpu', 'gpu%d' % x)
            for x in range(self._num_devices)
        ]

    def _gen_dtod_command_settings(self):
        """Generate device-to-device command settings.

        Return:
            Device-to-device command settings.
        """
        return [
            self._gen_command_setting('dtod_gpu%d' % x, 'gpu%d' % x, 'gpu%d' % x, 'gpu%d' % x)
            for x in range(self._num_devices)
        ]

    def _gen_ptop_command_settings(self):
        """Generate peer-to-peer command settings.

        Return:
            Peer-to-peer command settings.
        """
        command_settings = []
        for x in range(self._num_devices):
            for y in range(self._num_devices):
                command_settings.append(
                    self._gen_command_setting(
                        'ptop_gpu%d_reads_gpu%d' % (y, x), 'gpu%d' % x, 'gpu%d' % y, 'gpu%d' % y
                    )
                )
                command_settings.append(
                    self._gen_command_setting(
                        'ptop_gpu%d_writes_gpu%d' % (x, y), 'gpu%d' % x, 'gpu%d' % y, 'gpu%d' % x
                    )
                )
        return command_settings

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        if self._num_devices == 0:
            self._commands = []
            return True

        for mem_type in self._args.mem_type:
            if mem_type == 'htoh':
                self._command_settings += self._gen_htoh_command_settings()
            elif mem_type == 'htod':
                self._command_settings += self._gen_htod_command_settings()
            elif mem_type == 'dtoh':
                self._command_settings += self._gen_dtoh_command_settings()
            elif mem_type == 'dtod':
                self._command_settings += self._gen_dtod_command_settings()
            elif mem_type == 'ptop':
                self._command_settings += self._gen_ptop_command_settings()
            else:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported mem_type - benchmark: {}, mem_type: {}, expected: {}.'.format(
                        self._name, mem_type, ' '.join(self._mem_types)
                    )
                )
                return False

        self._commands = [
            '%s %s %d %d' % (self.__bin_path, ' '.join(x['devices']), self._args.size, self._args.num_loops)
            for x in self._command_settings
        ]

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
