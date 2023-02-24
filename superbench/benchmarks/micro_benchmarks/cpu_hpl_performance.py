# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module for running the HPL benchmark."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CpuHplBenchmark(MicroBenchmarkWithInvoke):
    """The HPL benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'hpl_run.sh'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self.__cpu_arch = ['zen3', 'zen4']

        self._parser.add_argument(
            '--cpu_arch',
            type=str,
            default='zen4',
            required=False,
            help='The targeted cpu architectures to run \
                HPL. Default is zen4. Possible values are {}.'.format(' '.join(self.__cpu_arch))
        )
        self._parser.add_argument(
            '--blockSize',
            type=int,
            default=384,
            required=False,
            help='Size of blocks. This parameter is an HPL input. Default 384.'
        )
        self._parser.add_argument(
            '--coreCount',
            type=int,
            default=88,    # for HBv4 total number of cores is 176 => 88 per cpu
            required=False,
            help='Number of cores per CPU. Used for MPI and HPL configuration. \
            Default 88 (HBv4 has a total of 176 w/ 2 cpus therefore 88 per cpu)'
        )
        self._parser.add_argument(
            '--blocks',
            type=int,
            default=1,
            required=False,
            help='Number of blocks. This parameter is an HPL input. Default 1.'
        )
        self._parser.add_argument(
            '--problemSize',
            type=int,
            default=384000,
            required=False,
            help='This is the problem size designated by "N" notation. \
            This parameter is an HPL input. Default is 384000'
        )

    def _preprocess(self, hpl_template):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if not self._set_binary_path():
            logger.error(
                'Executable {} not found in {} or it is not executable'.format(self._bin_name, self._args.bin_dir)
            )
            return False

        # xhpl type
        xhpl = 'xhpl_z4'
        if self._args.cpu_arch == 'zen3':
            xhpl = 'xhpl_z3'

        # command
        command = os.path.join(self._args.bin_dir, self._bin_name)
        command = command + ' ' + xhpl + ' ' + str(self._args.coreCount)

        # modify HPL.dat
        if hpl_template:
            hpl_input_file = hpl_template
        else:
            hpl_input_file = os.path.join(self._args.bin_dir, 'template_hpl.dat')
        search_string = ['problemSize', 'blockCount', 'blockSize']
        with open(hpl_input_file, 'r') as hplfile:
            lines = hplfile.readlines()
        hpl_input_file = os.path.join(os.getcwd(), 'HPL.dat')
        with open(hpl_input_file, 'w') as hplfile:
            for line in lines:
                if search_string[0] in line:
                    line = line.replace(search_string[0], str(self._args.problemSize))
                elif search_string[1] in line:
                    line = line.replace(search_string[1], str(self._args.blocks))
                elif search_string[2] in line:
                    line = line.replace(search_string[2], str(self._args.blockSize))
                hplfile.write(line)

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
        content = raw_output.splitlines()

        for idx, line in enumerate(content):
            if 'T/V' in line and 'Gflops' in line:
                break

        results = content[idx + 2].split()

        for line in content[idx + 2:]:
            if '1 tests completed and passed residual checks' in line:
                self._result.add_result('tests_pass', 1)
            elif '0 tests completed and passed residual checks' in line:
                self._result.add_result('tests_pass', 0)

        self._result.add_result('time', float(results[5]))
        self._result.add_result('throughput', float(results[6]))

        # raw output
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        return True


BenchmarkRegistry.register_benchmark('cpu-hpl', CpuHplBenchmark)
