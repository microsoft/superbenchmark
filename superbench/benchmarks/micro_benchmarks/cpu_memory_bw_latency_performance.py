# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module to measure memory bandwidth and latency."""

import os
import platform

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CpuMemBwLatencyBenchmark(MicroBenchmarkWithInvoke):
    """The Memory bandwidth and latency benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'mlc' if 'x86_64' in platform.machine() else 'cpu_copy'
        self.__support_mlc_commands = ['bandwidth_matrix', 'latency_matrix', 'max_bandwidth']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        # Add arguments for the Intel MLC tool.
        self._parser.add_argument(
            '--tests',
            type=str,
            nargs='+',
            default=['bandwidth_matrix'],
            required=False,
            help='The modes to run mlc with. Possible values are {}.'.format(' '.join(self.__support_mlc_commands))
        )

        # Add arguments for the general CPU copy benchmark.
        self._parser.add_argument(
            '--size',
            type=int,
            default=256 * 1024**2,
            required=False,
            help='Size of data buffer in bytes for non mlc benchmark. Default is 256MB.',
        )

        self._parser.add_argument(
            '--num_warm_up',
            type=int,
            default=20,
            required=False,
            help='Number of warm up rounds for non mlc benchmark. Default is 20.',
        )

        self._parser.add_argument(
            '--num_loops',
            type=int,
            default=100,
            required=False,
            help='Number of data buffer copies performed for non mlc benchmark. Default is 100.',
        )

        self._parser.add_argument(
            '--check_data',
            action='store_true',
            help='Enable data checking for non mlc benchmark. Default is False.',
        )

    def _preprocess_mlc(self):
        """Preprocess/preparation operations for the Intel MLC tool."""
        mlc_path = os.path.join(self._args.bin_dir, self._bin_name)
        ret_val = os.access(mlc_path, os.X_OK | os.F_OK)
        if not ret_val:
            logger.error(
                'Executable {} not found in {} or it is not executable'.format(self._bin_name, self._args.bin_dir)
            )
            return False

        # the mlc command requires hugapage to be enabled
        mlc_wrapper = ' '.join(
            [
                'nr_hugepages=`cat /proc/sys/vm/nr_hugepages`;', 'echo 4000 > /proc/sys/vm/nr_hugepages;', '%s;',
                'err=$?;', 'echo ${nr_hugepages} > /proc/sys/vm/nr_hugepages;', '(exit $err)'
            ]
        )
        for test in self._args.tests:
            command = mlc_path + ' --%s' % test
            self._commands.append(mlc_wrapper % command)
        return True

    def _preprocess_general(self):
        """Preprocess/preparation operations for the general CPU copy benchmark."""
        # TODO: enable hugepages?

        self.__bin_path = os.path.join(self._args.bin_dir, self._bin_name)

        args = '--size %d --num_warm_up %d --num_loops %d' % (
            self._args.size, self._args.num_warm_up, self._args.num_loops
        )

        if self._args.check_data:
            args += ' --check_data'

        self._commands = ['%s %s' % (self.__bin_path, args)]

        return True

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        return self._preprocess_mlc() if 'x86_64' in platform.machine() else self._preprocess_general()

    def _process_raw_result_mlc(self, cmd_idx, raw_output):
        """Function to parse raw results for the Intel MLC tool and save the summarized results."""
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        # parse the command to see which command this output belongs to
        # the command is formed as ...; mlc --option; ...
        # option needs to be extracted
        if '--' in self._commands[cmd_idx]:
            mlc_test = self._commands[cmd_idx].split('--')[1]
        else:
            logger.error('The command {} is not well formed and missing --'.format(self._commands[cmd_idx]))
            return False
        mlc_test = mlc_test.split(';')[0]
        if 'max_bandwidth' in mlc_test:
            measure = 'bw'
            out_table = self._parse_max_bw(raw_output)
        elif 'bandwidth_matrix' in mlc_test:
            measure = 'bw'
            out_table = self._parse_bw_latency(raw_output)
        elif 'latency_matrix' in mlc_test:
            measure = 'lat'
            out_table = self._parse_bw_latency(raw_output)
        else:
            logger.error('Invalid option {} to run the {} command'.format(mlc_test, self._commands[cmd_idx]))
            return False
        if len(out_table) == 0:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, raw_output
                )
            )
            return False
        for key in out_table.keys():
            for index in range(len(out_table[key])):
                if 'max_bandwidth' in mlc_test:
                    metric = 'mem_{}_{}_{}'.format(mlc_test, key, measure).lower()
                else:
                    metric = 'mem_{}_{}_{}_{}'.format(mlc_test, key, str(index), measure).lower()
                self._result.add_result(metric, float(out_table[key][index]))

        return True

    def _process_raw_result_general(self, cmd_idx, raw_output):
        """Function to parse raw results for the general CPU copy benchmark and save the summarized results."""
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        try:
            for output_line in raw_output.strip().splitlines():
                name, value = output_line.split(':')
                self._result.add_result(name.strip(), float(value.strip()))
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )

            return False

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
        return (
            self._process_raw_result_mlc(cmd_idx, raw_output)
            if 'x86_64' in platform.machine() else self._process_raw_result_general(cmd_idx, raw_output)
        )

    def _parse_bw_latency(self, raw_output):
        out_table = dict()
        for line in raw_output.splitlines():
            if line.strip() == '':
                continue
            # only lines starting with a digit is of interest
            if line.lstrip()[0].isdigit():
                vals = line.split()
                if len(vals) < 2:
                    continue
                numa_index = 'numa_%s' % vals[0]
                out_table[numa_index] = vals[1:]
        return out_table

    def _parse_max_bw(self, raw_output):
        out_table = dict()
        # the very last line is empty and only the last 5 lines of the output are of interest
        for line in raw_output.splitlines()[-6:]:
            if line.strip() == '':
                continue
            vals = line.split()
            if len(vals) < 2:
                continue
            key = '_'.join(vals[0:2]).rstrip(':').replace(':', '_')
            # making a list to be consistent with the _parse_bw_latency output
            out_table[key] = [vals[-1]]
        return out_table


BenchmarkRegistry.register_benchmark('cpu-memory-bw-latency', CpuMemBwLatencyBenchmark)
