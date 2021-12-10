# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the IB performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.common.utils import network
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.common.devices import GPU
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class IBBenchmark(MicroBenchmarkWithInvoke):
    """The IB validation performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'ib_validation'
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
        """Generate one-to-many pattern config.

        There are a total of n rounds
        In each round, The i-th participant will be paired as a client with the remaining n-1 servers.

        Args:
            n (int): the number of participants.

        Returns:
            list: the generated config list, each item in the list is a str like "0,1;2,3".
        """
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
        """Generate many-to-one pattern config.

        There are a total of n rounds
        In each round, The i-th participant will be paired as a server with the remaining n-1 clients.

        Args:
            n (int): the number of participants.

        Returns:
            list: the generated config list, each item in the list is a str like "0,1;2,3".
        """
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

    def __fully_one_to_one(self, n):
        """Generate one-to-one pattern config.

        One-to-one means that each participant plays every other participant once.
        The algorithm refers circle method of Round-robin tournament in
        https://en.wikipedia.org/wiki/Round-robin_tournament.
        if n is even, there are a total of n-1 rounds, with n/2 pair of 2 unique participants in each round.
        If n is odd, there will be n rounds, each with n-1/2 pairs, and one participant rotating empty in that round.
        In each round, pair up two by two from the beginning to the middle as (begin, end),(begin+1,end-1)...
        Then, all the participants except the beginning shift left one position, and repeat the previous step.

        Args:
            n (int): the number of participants.

        Returns:
            list: the generated config list, each item in the list is a str like "0,1;2,3".
        """
        config = []
        candidates = list(range(n))
        # Add a fake participant if n is odd
        if n % 2 == 1:
            candidates.append(-1)
        count = len(candidates)
        non_moving = [candidates[0]]
        for _ in range(count - 1):
            pairs = [
                '{},{}'.format(candidates[i], candidates[count - i - 1]) for i in range(0, count // 2)
                if candidates[i] != -1 and candidates[count - i - 1] != -1
            ]
            row = ';'.join(pairs)
            config.append(row)
            robin = candidates[2:] + candidates[1:2]
            candidates = non_moving + robin
        return config

    def gen_traffic_pattern(self, n, mode, config_file_path):
        """Generate traffic pattern into config file.

        Args:
            n (int): the number of nodes.
            mode (str): the traffic mode, including 'one-to-one', 'one-to-many', 'many-to-one'.
            config_file_path (str): the path of config file to generate.
        """
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

    def __prepare_config(self, node_num):
        """Prepare and read config file.

        Args:
            node_num (int): the number of nodes.

        Returns:
            True if the config is not empty and valid.
        """
        try:
            # Generate the config file if not define
            if self._args.config is None:
                self.gen_traffic_pattern(node_num, self._args.pattern, self.__config_path)
            # Use the config file defined in args
            else:
                self.__config_path = self._args.config
            # Read the hostfile
            with open(self._args.hostfile, 'r') as f:
                hosts = f.readlines()
            # Read the config file and check if it's empty and valid
            with open(self.__config_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                pairs = line.strip().strip(';').split(';')
                # Check format of config
                for pair in pairs:
                    pair = pair.split(',')
                    if len(pair) != 2:
                        return False
                    pair[0] = int(pair[0])
                    pair[1] = int(pair[1])
                    self.__config.append('{}_{}'.format(hosts[pair[0]].strip(), hosts[pair[1]].strip()))
        except BaseException as e:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Failed to generate and check config - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False
        if len(self.__config) == 0:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('No valid config - benchmark: {}.'.format(self._name))
            return False
        return True

    def __prepare_general_ib_command_params(self):
        """Prepare general params for ib commands.

        Returns:
            Str of ib command params if arguments are valid, otherwise False.
        """
        # Format the ib command type
        self._args.commands = [command.lower() for command in self._args.commands]
        # Add message size for ib command
        msg_size = ''
        if self._args.msg_size is None:
            msg_size = '-a'
        else:
            msg_size = '-s ' + str(self._args.msg_size)
        # Add GPUDirect for ib command
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
        # Generate ib command params
        try:
            command_params = '-F --iters={iter} -d {device} {size}{gpu}'.format(
                iter=str(self._args.iters),
                device=network.get_ib_devices()[self._args.ib_index].split(':')[0],
                size=msg_size,
                gpu=gpu_enable
            )
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
            logger.error('Getting ib devices failure - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return False
        return command_params

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
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_MPI_INIT_FAILURE)
            logger.error('No MPI environment - benchmark: {}.'.format(self._name))
            return False

        # Generate and check config
        if not self.__prepare_config(node_num):
            return False

        # Prepare general params for ib commands
        command_params = self.__prepare_general_ib_command_params()
        if not command_params:
            return False
        # Generate commands
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

    def _process_raw_result(self, cmd_idx, raw_output):    # noqa: C901
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
        command = self._args.commands[cmd_idx]
        suffix = command.split('_')[-1]
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
                    pair_index = 0
                    for item in line:
                        metric = '{command}_{line}_{pair}_{host}_{suffix}'.format(
                            command=command,
                            line=str(line_index),
                            pair=pair_index,
                            host=self.__config[config_index],
                            suffix=suffix
                        )
                        value = float(item)
                        if 'bw' in command:
                            value = value / 1000
                        self._result.add_result(metric, value)
                        valid = True
                        config_index += 1
                        pair_index += 1
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
