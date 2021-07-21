# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Disk Performance benchmarks."""

from pathlib import Path
import json
import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class DiskPerformance(MicroBenchmarkWithInvoke):
    """The KernelLaunch overhead benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'fio'

        self._io_patterns = ['seq', 'rand']
        self._io_types = ['read', 'write']
        self._rand_block_size = 4 * 1024    # 4KiB
        self._seq_block_size = 128 * 1024    # 128KiB
        self._default_iodepth = 64
        self._default_ramp_time = 10
        self._default_runtime = 60
        self._default_numjobs_for_rand = 4

        self._common_fio_args =\
            ' --randrepeat=1 --thread=1 --ioengine=libaio --direct=1'\
            ' --norandommap=1 --lat_percentiles=1 --group_reporting=1'\
            ' --output-format=json'
        self._fio_args = {}

        # Sequentially write 128KiB to the device twice
        self._fio_args['seq_precond'] = self._common_fio_args +\
            ' --name=seq_precond --rw=write --bs=%d --iodepth=%d --numjobs=1 --loops=2' %\
            (self._seq_block_size, self._default_iodepth)

        # Randomly write 4KiB to the device
        self._fio_args['rand_precond'] = self._common_fio_args +\
            ' --name=rand_precond --rw=randwrite --bs=%d --iodepth=%d --numjobs=%d --time_based=1' %\
            (self._rand_block_size, self._default_iodepth, self._default_numjobs_for_rand)

        # Seq/rand read/write tests
        for io_pattern in self._io_patterns:
            for io_type in self._io_types:
                io_str = '%s_%s' % (io_pattern, io_type)
                fio_rw = io_type if io_pattern == 'seq' else io_pattern + io_type
                fio_bs = self._seq_block_size if io_pattern == 'seq' else self._rand_block_size
                self._fio_args[io_str] = self._common_fio_args +\
                    ' --name=%s --rw=%s --bs=%d --time_based=1' % (io_str, fio_rw, fio_bs)

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--block_devices',
            type=str,
            nargs='*',
            default=[],
            required=False,
            help='Disk block device(s) to be tested.',
        )

        # Disable precondition by default
        self._parser.add_argument(
            '--enable_seq_precond',
            type=int,
            default=0,
            required=False,
            help='Enable seq write precondition.',
        )
        self._parser.add_argument(
            '--rand_precond_time',
            type=int,
            default=0,
            required=False,
            help='Time in seconds to run rand write precondition. Set to 0 to disable this test.',
        )

        for io_pattern in self._io_patterns:
            for io_type in self._io_types:
                io_str = '%s_%s' % (io_pattern, io_type)
                self._parser.add_argument(
                    '--%s_ramp_time' % io_str,
                    type=int,
                    default=self._default_ramp_time,
                    required=False,
                    help='Time in seconds to warm up %s test.' % io_str,
                )
                # Disable write tests by default
                default_runtime = 0 if io_type == 'write' else self._default_runtime
                self._parser.add_argument(
                    '--%s_runtime' % io_str,
                    type=int,
                    default=default_runtime,
                    required=False,
                    help='Time in seconds to run %s test. Set to 0 to disable this test.' % io_str,
                )
                self._parser.add_argument(
                    '--%s_iodepth' % io_str,
                    type=int,
                    default=self._default_iodepth,
                    required=False,
                    help='Queue depth for each thread in %s test.' % io_str,
                )
                default_numjobs = 1 if io_pattern == 'seq' else self._default_numjobs_for_rand
                self._parser.add_argument(
                    '--%s_numjobs' % io_str,
                    type=int,
                    default=default_numjobs,
                    required=False,
                    help='Number of threads in %s test.' % io_str,
                )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        fio_path = os.path.join(self._args.bin_dir, self._bin_name)

        for block_device in self._args.block_devices:
            if not Path(block_device).is_block_device():
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('Invalid block device: {}.'.format(block_device))
                return False

            if self._args.enable_seq_precond:
                command = fio_path +\
                    ' --filename=%s' % block_device +\
                    self._fio_args['seq_precond']
                self._commands.append(command)

            if self._args.rand_precond_time > 0:
                command = fio_path +\
                    ' --filename=%s' % block_device +\
                    ' --runtime=%ds' % self._args.rand_precond_time +\
                    self._fio_args['rand_precond']
                self._commands.append(command)

            for io_pattern in self._io_patterns:
                for io_type in self._io_types:
                    io_str = '%s_%s' % (io_pattern, io_type)
                    runtime = getattr(self._args, '%s_runtime' % io_str)
                    if runtime > 0:
                        command = fio_path +\
                            ' --filename=%s' % block_device +\
                            ' --ramp_time=%ds' % getattr(self._args, '%s_ramp_time' % io_str) +\
                            ' --runtime=%ds' % runtime +\
                            ' --iodepth=%d' % getattr(self._args, '%s_iodepth' % io_str) +\
                            ' --numjobs=%d' % getattr(self._args, '%s_numjobs' % io_str) +\
                            self._fio_args[io_str]
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
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output)

        try:
            fio_output = json.loads(raw_output)
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False

        jobname = fio_output['jobs'][0]['jobname']
        block_device = fio_output['global options']['filename']
        jobname_prefix = 'disk_performance:%s:%s' % (block_device, jobname)
        lat_units = ['lat_ns', 'lat_us', 'lat_ms']

        bs = fio_output['jobs'][0]['job options']['bs']
        self._result.add_result('%s:bs' % jobname_prefix, float(bs))

        for io_type in ['read', 'write']:
            io_type_prefix = '%s:%s' % (jobname_prefix, io_type)

            iops = fio_output['jobs'][0][io_type]['iops']
            self._result.add_result('%s:iops' % io_type_prefix, float(iops))

            for lat_unit in lat_units:
                if lat_unit in fio_output['jobs'][0][io_type]:
                    lat_unit_prefix = '%s:%s' % (io_type_prefix, lat_unit)
                    for lat_percentile in ['95.000000', '99.000000', '99.900000']:
                        lat = fio_output['jobs'][0][io_type][lat_unit]['percentile'][lat_percentile]
                        self._result.add_result('%s:%s' % (lat_unit_prefix, lat_percentile), float(lat))
                    break

        return True


BenchmarkRegistry.register_benchmark('disk-performance', DiskPerformance)
