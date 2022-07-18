# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the IB loopback benchmarks."""

import os
import socket
from pathlib import Path

from superbench.common.utils import logger
from superbench.common.utils import network
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


def get_numa_cores(numa_index):
    """Get the available cores from different physical cpu core of NUMA<numa_index>.

    Args:
        numa_index (int): the index of numa node.

    Return:
        list: The available cores from different physical cpu core of NUMA<numa_index>.
        None if no available cores or numa index.
    """
    try:
        with Path(f'/sys/devices/system/node/node{numa_index}/cpulist').open('r') as f:
            cores = []
            core_ranges = f.read().strip().split(',')
            for core_range in core_ranges:
                start, end = core_range.split('-')
                for core in range(int(start), int(end) + 1):
                    cores.append(core)
        return cores
    except IOError:
        return None


class IBLoopbackBenchmark(MicroBenchmarkWithInvoke):
    """The IB loopback performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'run_perftest_loopback'
        self.__sock_fds = []
        self.__support_ib_commands = {'write': 'ib_write_bw', 'read': 'ib_read_bw', 'send': 'ib_send_bw'}

    def __del__(self):
        """Destructor."""
        for fd in self.__sock_fds:
            fd.close()

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--ib_index',
            type=int,
            default=0,
            required=False,
            help='The index of ib device.',
        )
        self._parser.add_argument(
            '--iters',
            type=int,
            default=20000,
            required=False,
            help='The iterations of running ib command',
        )
        self._parser.add_argument(
            '--msg_size',
            type=int,
            default=None,
            required=False,
            help='The message size of running ib command, e.g., 8388608.',
        )
        self._parser.add_argument(
            '--commands',
            type=str,
            nargs='+',
            default=['write'],
            help='The ib command used to run, e.g., {}.'.format(' '.join(list(self.__support_ib_commands.keys()))),
        )
        self._parser.add_argument(
            '--numa',
            type=int,
            default=0,
            required=False,
            help='The index of numa node.',
        )
        self._parser.add_argument(
            '--gid_index',
            type=int,
            default=0,
            required=False,
            help='Test uses GID with GID index taken from command.',
        )

    def __get_arguments_from_env(self):
        """Read environment variables from runner used for parallel and fill in ib_index and numa_node_index.

        Get 'PROC_RANK'(rank of current process) 'IB_DEVICES' 'NUMA_NODES' environment variables
        Get ib_index and numa_node_index according to 'NUMA_NODES'['PROC_RANK'] and 'IB_DEVICES'['PROC_RANK']
        Note: The config from env variables will overwrite the configs defined in the command line
        """
        try:
            if os.getenv('PROC_RANK'):
                rank = int(os.getenv('PROC_RANK'))
                if os.getenv('IB_DEVICES'):
                    self._args.ib_index = int(os.getenv('IB_DEVICES').split(',')[rank])
                if os.getenv('NUMA_NODES'):
                    self._args.numa = int(os.getenv('NUMA_NODES').split(',')[rank])
            return True
        except BaseException:
            logger.error('The proc_rank is out of index of devices - benchmark: {}.'.format(self._name))
            return False

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess() or not self.__get_arguments_from_env():
            return False

        # Format the arguments
        self._args.commands = [command.lower() for command in self._args.commands]

        # Check whether arguments are valid
        command_mode = ''
        if self._args.msg_size is None:
            command_mode = ' -a'
        else:
            command_mode = ' -s ' + str(self._args.msg_size)

        for ib_command in self._args.commands:
            if ib_command not in self.__support_ib_commands:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported ib command - benchmark: {}, command: {}, expected: {}.'.format(
                        self._name, ib_command, ' '.join(list(self.__support_ib_commands.keys()))
                    )
                )
                return False

            try:
                self.__sock_fds.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                # grep SO_REUSE /usr/include/asm-generic/socket.h
                self.__sock_fds[-1].setsockopt(socket.SOL_SOCKET, getattr(socket, 'SO_REUSEADDR', 2), 1)
                self.__sock_fds[-1].setsockopt(socket.SOL_SOCKET, getattr(socket, 'SO_REUSEPORT', 15), 1)
                self.__sock_fds[-1].bind(('127.0.0.1', 0))
            except OSError as e:
                self._result.set_return_code(ReturnCode.RUNTIME_EXCEPTION_ERROR)
                logger.error('Error when binding port - benchmark: %s, message: %s.', self._name, e)
                return False
            try:
                ib_devices = network.get_ib_devices()
            except BaseException as e:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
                logger.error('Getting ib devices failure - benchmark: {}, message: {}.'.format(self._name, str(e)))
                return False
            numa_cores = get_numa_cores(self._args.numa)
            if not numa_cores or len(numa_cores) < 2:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
                logger.error('Getting numa core devices failure - benchmark: {}.'.format(self._name))
                return False
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' ' + str(numa_cores[-1]) + ' ' + str(numa_cores[-3 + int((len(numa_cores) < 4))])
            command += ' ' + os.path.join(self._args.bin_dir, self.__support_ib_commands[ib_command])
            command += command_mode + ' -F'
            command += ' --iters=' + str(self._args.iters)
            command += ' -d ' + ib_devices[self._args.ib_index].split(':')[0]
            command += ' -p ' + str(self.__sock_fds[-1].getsockname()[1])
            command += ' -x ' + str(self._args.gid_index)
            command += ' --report_gbits'
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
        self._result.add_raw_data(
            'raw_output_' + self._args.commands[cmd_idx] + '_IB' + str(self._args.ib_index), raw_output,
            self._args.log_raw_data
        )

        valid = False
        content = raw_output.splitlines()

        metric_set = set()
        for line in content:
            try:
                values = list(filter(None, line.split()))
                if len(values) != 5:
                    continue
                # Extract value from the line
                size = int(values[0])
                avg_bw = float(values[-2]) / 8.0
                metric = f'{self.__support_ib_commands[self._args.commands[cmd_idx]]}_{size}:{self._args.ib_index}'
                # Filter useless value in client output
                if metric not in metric_set:
                    metric_set.add(metric)
                    self._result.add_result(metric, avg_bw)
                    valid = True
            except BaseException:
                pass
        if valid is False:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, raw_output
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('ib-loopback', IBLoopbackBenchmark)
