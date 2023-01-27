# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module for running the University of Virginia STREAM tool. It measures sustainable main memory
    bandwidth in MB/s and the corresponding computation rate for simple vector kernels."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke

class CpuStreamBenchmark(MicroBenchmarkWithInvoke):
    """The Stream benchmark class."""

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'stream'
        self.__cpu_arch = ['zen1', 'zen2', 'zen3','zen4']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--CPU_ARCH',
            type=str,
            default=['other'],
            required=False,
            help='The targeted cpu architectures to run STREAM. Possible values are {}.'.format(' '.join(self.__cpu_arch))
        )
        self._parser.add_argument(
            '--cores',
            nargs='+'
            type=int,
            required=True,
            help='List of cores to perform test'
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False
        if not self._set_binary_path():
            return False


        # zen3
        # cores=[0, 4, 8, 12, 16, 20, 24, 28, 30, 34, 38, 42, 46, 50, 54, 58, 60, 64, 68, 72, 76, 80, 84, 88, 90, 94, 98, 102, 106, 110, 114, 118]
        # zen4
        # cores=[0, 8, 16, 24, 32, 38, 44, 52, 60, 68, 76, 82, 88, 96, 104, 112, 120, 126, 132, 140, 148, 156, 164, 170]
        
        omp_places=''
        for  in self._args.cores:
            omp_places+='{'+'{}:1'.format(core)+'}'

        envar='OMP_SCHEDULE=static && OMP_DYNAMIC=false && OMP_MAX_ACTIVE_LEVELS=1 && OMP_STACKSIZE=256M OMP_PROC_BIND=true && OMP_NUM_THREADS={} && OMP_PLACES={}'.format( self._args.cores.len, omp_places)
        if self._args.CPU_ARCH == 'other':
            return False
        elif self._args.CPU_ARCH == 'zen3':        
            exe='streamZen3.exe'
        elif self._args.CPU_ARCH == 'zen4'
            exe='streamZen4.exe'
        command =envar + " " + os.path.join(self._args.bin_dir, exe)
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
        #individual results
        self._result.add_result(metric, float(out_table[key][index]))
        #raw output
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)
        #content = raw_output.splitlines()
        return




BenchmarkRegistry.register_benchmark('cpu-stream', CpuStreamBenchmark)
