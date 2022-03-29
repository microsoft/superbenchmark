# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module for running the Intel MLC tool to measure memory bandwidth and latency."""

import os

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

        self._bin_name = 'mlc'
        self.__support_mlc_commands = ['bandwidth_matrix', 'latency_matrix', 'max_bandwidth']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--tests',
            type=str,
            nargs='+',
            default=['bandwidth_matrix'],
            required=False,
            help='The modes to run mlc with. Possible values are {}.'.format(' '.join(self.__support_mlc_commands))
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

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
