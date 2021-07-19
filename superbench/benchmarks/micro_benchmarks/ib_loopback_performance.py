# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the IB loopback benchmarks."""

import os
import subprocess

from superbench.common.utils import logger
from superbench.common.utils import network
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


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
        self.__support_ib_commands = {'write': 'ib_write_bw', 'read': 'ib_read_bw', 'send': 'ib_send_bw'}

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
            '--n',
            type=int,
            default=20000,
            required=False,
            help='The iterations of running ib command',
        )
        self._parser.add_argument(
            '--size',
            type=int,
            default=8388608,
            required=False,
            help='The message size of running ib command. E.g. 8388608.',
        )
        self._parser.add_argument(
            '--commands',
            type=str,
            nargs='+',
            default='write',
            help='The ib command used to run. E.g. {}.'.format(' '.join(list(self.__support_ib_commands.keys()))),
        )
        self._parser.add_argument(
            '--mode',
            type=str,
            default='AF',
            help='The mode used to run ib command. Eg, AF(all message size) or S(single message size)',
        )
        self._parser.add_argument(
            '--numa',
            type=int,
            default=0,
            required=False,
            help='The index of numa node.',
        )
        self._parser.add_argument(
            '--x',
            type=int,
            default=0,
            required=False,
            help='Test uses GID with GID index taken from command.',
        )

    def __get_numa_cores(self, numa_index):
        """Get the last two cores from different physical cpu core of NUMA<numa_index>.

        Args:
            numa_index (int): the index of numa node.

        Return:
            The last two cores from different physical cpu core of NUMA<numa_index>.
        """
        command = 'numactl --hardware | grep "node {} cpus:"'.format(numa_index)
        output = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, check=False, universal_newlines=True
        )
        return output.stdout.splitlines()[0].split(' ')

    def __get_arguments_from_env(self):
        """Read environment variables from runner used for parallel and fill in ib_index and numa_node_index.

        Get 'PROC_RANK'(rank of current process) 'IB_DEVICES' 'NUMA_NODES' environment variables
        Get ib_index and numa_node_index according to 'NUMA_NODES'['PROC_RANK'] and 'IB_DEVICES'['PROC_RANK']
        """
        if os.getenv('PROC_RANK'):
            rank = int(os.getenv('PROC_RANK'))
            if os.getenv('IB_DEVICES'):
                self._args.ib_index = int(os.getenv('IB_DEVICES').split(',')[rank])
            if os.getenv('NUMA_NODES'):
                self._args.numa = int(os.getenv('NUMA_NODES').split(',')[rank])

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__get_arguments_from_env()

        # Format the arguments
        if not isinstance(self._args.commands, list):
            self._args.commands = [self._args.commands]
        self._args.commands = [command.lower() for command in self._args.commands]
        self._args.mode = self._args.mode.upper()

        # Check whether arguments are valid
        command_mode = ''
        if self._args.mode == 'AF':
            command_mode = ' -a'
        elif self._args.mode == 'S':
            command_mode = ' -s ' + str(self._args.size)
        else:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error(
                'Unsupported args mode - benchmark: {}, mode: {}, expect: {}.'.format(
                    self._name, self._args.mode, 'AF or S'
                )
            )
            return False

        for ib_command in self._args.commands:
            if ib_command not in self.__support_ib_commands:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported ib command - benchmark: {}, command: {}, expected: {}.'.format(
                        self._name, ib_command, ' '.join(list(self.__support_ib_commands.keys()))
                    )
                )
                return False
            else:
                try:
                    command = os.path.join(self._args.bin_dir, self._bin_name)
                    numa_cores = self.__get_numa_cores(self._args.numa)
                    server_core = int(numa_cores[-1])
                    client_core = int(numa_cores[-3])
                    command += ' ' + str(server_core) + ' ' + str(client_core)
                    command += ' ' + self.__support_ib_commands[ib_command]
                    command += command_mode + ' -F'
                    command += ' --iters=' + str(self._args.n)
                    command += ' -d ' + network.get_ib_devices()[self._args.ib_index]
                    command += ' -p ' + str(network.get_free_port())
                    command += ' -x ' + str(self._args.x)
                    self._commands.append(command)
                except BaseException as e:
                    self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
                    logger.error('Getting devices failure - benchmark: {}, message: {}.'.format(self._name, str(e)))
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
        self._result.add_raw_data(
            'raw_output_' + self._args.commands[cmd_idx] + '_IB' + str(self._args.ib_index), raw_output
        )

        valid = False
        content = raw_output.splitlines()
        try:
            metric_set = set()
            for line in content:
                if str(self._args.size) in line:
                    values = list(filter(None, line.split(' ')))
                    avg_bw = float(values[-2])
                    metric = 'IB_{}_Avg_{}'.format(self._args.commands[cmd_idx], str(self._args.ib_index))
                    # Filter useless value
                    if metric not in metric_set:
                        metric_set.add(metric)
                        self._result.add_result(metric, avg_bw)
                        valid = True
        except BaseException:
            valid = False
        finally:
            if valid is False:
                logger.error(
                    'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                        self._curr_run_index, self._name, raw_output
                    )
                )
                return False

        return True


BenchmarkRegistry.register_benchmark('ib-loopback', IBLoopbackBenchmark)
