# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the IB traffic pattern benchmarks."""

import os

from superbench.common.utils import logger
from superbench.common.utils import network
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.common.devices import GPU
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class IBBenchmark(MicroBenchmarkWithInvoke):
    """The IB traffic pattern performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'ib_mpi'
        self.__support_ib_commands = [
            'ib_write_bw', 'ib_read_bw', 'ib_send_bw', 'ib_write_lat', 'ib_read_lat', 'ib_send_lat'
        ]
        self.__patterns = ['one-to-one', 'one-to-many', 'many-to-one']
        self.__config_path = os.getcwd() + '/config.txt'
        self.__config = []

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
            default=5000,
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
            default=['ib_write_bw'],
            help='The ib command used to run, e.g., {}.'.format(' '.join(self.__support_ib_commands)),
        )
        self._parser.add_argument(
            '--gid_index',
            type=int,
            default=0,
            required=False,
            help='Test uses GID with GID index taken from command.',
        )
        self._parser.add_argument(
            '--pattern',
            type=str,
            default='one-to-one',
            required=False,
            help='Test IB traffic pattern type, e.g., {}.'.format(''.join(self.__patterns)),
        )
        self._parser.add_argument(
            '--config',
            type=str,
            default=None,
            required=False,
            help='The path of config file on the target machines',
        )
        self._parser.add_argument(
            '--bidirectional', action='store_true', default=False, help='Measure bidirectional bandwidth.'
        )
        self._parser.add_argument(
            '--gpu_index', type=int, default=None, required=False, help='Test Use GPUDirect with the gpu index.'
        )
        self._parser.add_argument(
            '--hostfile',
            type=str,
            default='/root/hostfile',
            required=False,
            help='The path of hostfile on the target machines',
        )

    def __one_to_many(self, n):
        config = []
        for client in range(n):
            row = []
            for server in range(n):
                if server != client:
                    pair = '{},{}'.format(server, client)
                    row.append(pair)
            row = ';'.join(row)
            config.append(row)
        return config

    def __many_to_one(self, n):
        config = []
        for server in range(n):
            row = []
            for client in range(n):
                if server != client:
                    pair = '{},{}'.format(server, client)
                    row.append(pair)
            row = ';'.join(row)
            config.append(row)
        return config

    def __roundRobin(self, candidates):
        res = []
        count = len(candidates)
        robin = candidates
        first = [candidates[0]]
        for _ in range(count - 1):
            pairs = [[robin[i], robin[count - i - 1]] for i in range(0, (count + 1) // 2)]
            res.append(pairs)
            robin = robin[1:]
            robin = first + robin[1:] + robin[:1]
        return res

    def __fully_one_to_one(self, n):
        config = []
        if n % 2 == 1:
            return config
        candidates = list(range(n))
        res = self.__roundRobin(candidates)
        for line in res:
            row = []
            for pair in line:
                row.append('{},{}'.format(pair[0], pair[1]))
            row = ';'.join(row)
            config.append(row)
        return config

    def gen_traffic_pattern(self, n, mode, config_file_path):
        config = []
        if mode == 'one-to-many':
            config = self.__one_to_many(n)
        elif mode == 'many-to-one':
            config = self.__many_to_one(n)
        elif mode == 'one-to-one':
            config = self.__fully_one_to_one(n)
        with open(config_file_path, 'w') as f:
            for line in config:
                f.write(line + '\n')

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False
        # Check MPI environment
        self._args.pattern = self._args.pattern.lower()
        if os.getenv('OMPI_COMM_WORLD_SIZE'):
            node_num = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        else:
            self._result.set_return_code(ReturnCode.MPI_INIT_FAILURE)
            logger.error('No MPI environment - benchmark: {}.'.format(self._name))
            return False
        # Generate and check config
        try:
            if self._args.config is None:
                self.gen_traffic_pattern(node_num, self._args.pattern, self.__config_path)
            else:
                self.__config_path = self._args.config
            with open(self.__config_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    pairs = line.strip().split(';')
                    self.__config.extend(pairs)
            if len(self.__config) == 0:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('No valid config - benchmark: {}.'.format(self._name))
                return False
        except BaseException as e:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Failed to generate and check config - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        # Format the arguments
        self._args.commands = [command.lower() for command in self._args.commands]
        try:
            # Check whether arguments are valid
            msg_size = ''
            if self._args.msg_size is None:
                msg_size = '-a'
            else:
                msg_size = '-s ' + str(self._args.msg_size)
            gpu_enable = ''
            if self._args.gpu_index:
                gpu = GPU()
                if gpu.vendor == 'nvidia':
                    gpu_enable = ' --use_cuda={gpu_index}'.format(gpu_index=str(self._args.gpu_index))
                elif gpu.vendor == 'amd':
                    gpu_enable = ' --use_rocm={gpu_index}'.format(gpu_index=str(self._args.gpu_index))
                else:
                    self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                    logger.error('No GPU found - benchmark: {}'.format(self._name))
                    return False
            command_params = '-F --iters={iter} -d {device} {size} -x {gid}{gpu}'.format(
                iter=str(self._args.iters),
                device=network.get_ib_devices()[self._args.ib_index].split(':')[0],
                size=msg_size,
                gid=str(self._args.gid_index),
                gpu=gpu_enable
            )
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
            logger.error('Getting ib devices failure - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False

        for ib_command in self._args.commands:
            if ib_command not in self.__support_ib_commands:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported ib command - benchmark: {}, command: {}, expected: {}.'.format(
                        self._name, ib_command, ' '.join(self.__support_ib_commands)
                    )
                )
                return False
            else:
                ib_command_prefix = '{command} {command_params}'.format(
                    command=ib_command, command_params=command_params
                )
                if 'bw' in ib_command and self._args.bidirectional:
                    ib_command_prefix += ' -b'

                command = os.path.join(self._args.bin_dir, self._bin_name)
                command += ' --hostfile ' + self._args.hostfile
                command += ' --cmd_prefix ' + '\"' + ib_command_prefix + '\"'
                command += ' --input_config ' + self.__config_path
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
        self._result.add_raw_data('raw_output_' + self._args.commands[cmd_idx], raw_output)

        # If it's invoked by MPI and rank is not 0, no result is expected
        if os.getenv('OMPI_COMM_WORLD_RANK'):
            rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            if rank > 0:
                return True

        valid = False
        content = raw_output.splitlines()
        line_index = 0
        config_index = 0
        try:
            result_index = -1
            for index, line in enumerate(content):
                if 'results' in line:
                    result_index = index + 1
                    break
            if result_index == -1:
                valid = False
            else:
                content = content[result_index:]
                for line in content:
                    line = list(filter(None, line.strip().split(',')))
                    for item in line:
                        metric = '{line}-{pair}'.format(line=str(line_index), pair=self.__config[config_index])
                        self._result.add_result(metric, float(item))
                        valid = True
                        config_index += 1
                    line_index += 1
        except Exception:
            valid = False
        if valid is False or config_index != len(self.__config):
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, raw_output
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('ib-traffic', IBBenchmark)
