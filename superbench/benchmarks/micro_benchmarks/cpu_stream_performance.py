# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module for running the University of Virginia STREAM tool. It measures sustainable main memory
    bandwidth in MB/s and the corresponding computation rate for simple vector kernels."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform
# , ReturnCode
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

        self._bin_name = 'streamZen3.exe'
        self.__cpu_arch = ['other', 'zen3', 'zen4']

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--CPU_ARCH',
            type=str,
            default=['other'],
            required=False,
            help='The targeted cpu architectures to run \
                STREAM. Possible values are {}.'.format(' '.join(self.__cpu_arch))
        )
        self._parser.add_argument(
            '--cores',
            nargs='+',
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
        # cores=[0, 4, 8, 12, 16, 20, 24, 28, 30, 34, 38, 42, 46, 50,
        # 54, 58, 60, 64, 68, 72, 76, 80, 84, 88, 90, 94, 98, 102, 106, 110, 114, 118]
        # zen4
        # cores=[0, 8, 16, 24, 32, 38, 44, 52, 60, 68, 76, 82, 88, 96, 104, 112, 120,
        # 126, 132, 140, 148, 156, 164, 170]

        # parse cores argument
        omp_places = ''
        for core in self._args.cores:
            omp_places += '{' + '{}:1'.format(core) + '}'

        envar = 'OMP_SCHEDULE=static && OMP_DYNAMIC=false && OMP_MAX_ACTIVE_LEVELS=1 && OMP_STACKSIZE=256M && \
            OMP_PROC_BIND=true && OMP_NUM_THREADS={} && OMP_PLACES={}'.format(len(self._args.cores), omp_places)

        if self._args.CPU_ARCH == 'zen3':
            exe = 'streamZen3.exe'
        elif self._args.CPU_ARCH == 'zen4':
            exe = 'streamZen4.exe'
        else:
            exe = 'streamx86.exe'

        command = envar + " " + os.path.join(self._args.bin_dir, exe)
        self._bin_name = exe
        stream_path = os.path.join(self._args.bin_dir, self._bin_name)
        ret_val = os.access(stream_path, os.X_OK | os.F_OK)
        if not ret_val:
            print("fail3")
            logger.error(
                'Executable {} not found in {} or it is not executable'.format(self._bin_name, self._args.bin_dir)
            )
            return False

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
        functions = ['Copy', 'Scale', 'Add', 'Triad']
        records = []
        content = raw_output.splitlines()
        for line in content:
            if "Number of Threads counted" in line:
                line.split("= ")[1]
                self._result.add_result("Threads", int(line.split("= ")[1]))
            for function in functions:
                if function in line:
                    records.append(line)

        # individual results
        for record in records:
            entries = record.split()
            metric = entries[0].strip().replace(':', '')
            self._result.add_result(metric + "_Throughput", float(entries[1].strip()))
            self._result.add_result(metric + "_AvgTime", float(entries[2].strip()))
            self._result.add_result(metric + "_MinTime", float(entries[3].strip()))
            self._result.add_result(metric + "_MaxTime", float(entries[4].strip()))

        # raw output
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        return True


BenchmarkRegistry.register_benchmark('cpu-stream', CpuStreamBenchmark)
