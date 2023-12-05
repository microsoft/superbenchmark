# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the IB performance benchmarks."""

import os

from superbench.common.utils import logger
from superbench.common.utils import gen_pair_wise_config, gen_topo_aware_config
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
        self.__support_directions = ['gpu-to-gpu', 'cpu-to-cpu', 'cpu-to-gpu', 'gpu-to-cpu']
        self.__patterns = ['one-to-one', 'one-to-many', 'many-to-one', 'topo-aware']
        self.__config_path = os.path.join(os.getcwd(), 'config.txt')
        self.__config = []

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--ib_dev',
            type=str,
            default='mlx5_0',
            required=False,
            help='The IB device, e.g., mlx5_0, mlx5_$LOCAL_RANK, mlx5_$((LOCAL_RANK/2)), etc.',
        )
        self._parser.add_argument(
            '--gpu_dev',
            type=str,
            default=None,
            required=False,
            help='The GPU device, e.g., 0, $LOCAL_RANK, $((LOCAL_RANK/2)), etc.',
        )
        self._parser.add_argument(
            '--numa_dev',
            type=str,
            default=None,
            required=False,
            help='The NUMA node to bind, e.g., 0, $LOCAL_RANK, $((LOCAL_RANK/2)), etc.',
        )
        self._parser.add_argument(
            '--timeout',
            type=int,
            default=120,
            required=False,
            help='Timeout in seconds for each perftest command in case ib is too slow.',
        )
        # perftest configurations
        self._parser.add_argument(
            '--iters',
            type=int,
            default=5000,
            required=False,
            help='The iterations of perftest command',
        )
        self._parser.add_argument(
            '--msg_size',
            type=int,
            nargs='+',
            default=8388608,
            required=False,
            help='The message size of perftest command, e.g., 8388608.',
        )
        self._parser.add_argument(
            '--bidirectional', action='store_true', default=False, help='Measure bidirectional bandwidth.'
        )
        self._parser.add_argument(
            '--command',
            type=str,
            nargs='+',
            default='ib_write_bw',
            required=False,
            help='The perftest command to use, e.g., {}.'.format(' '.join(self.__support_ib_commands)),
        )
        # customized configurations
        self._parser.add_argument(
            '--pattern',
            type=str,
            default='one-to-one',
            help='IB traffic pattern type, e.g., {}.'.format(''.join(self.__patterns)),
        )
        self._parser.add_argument(
            '--config',
            type=str,
            default=None,
            required=False,
            help='The path of config file on the target machines.',
        )
        self._parser.add_argument(
            '--hostfile',
            type=str,
            default=None,
            required=False,
            help='The path of hostfile on the target machines.',
        )
        self._parser.add_argument(
            '--min_dist',
            type=int,
            default=2,
            required=False,
            help='The minimum distance of VM pair in topo-aware pattern',
        )
        self._parser.add_argument(
            '--max_dist',
            type=int,
            default=6,
            required=False,
            help='The maximum distance of VM pair in topo-aware pattern',
        )
        self._parser.add_argument(
            '--ibstat',
            type=str,
            default=None,
            required=False,
            help='The path of ibstat output',
        )
        self._parser.add_argument(
            '--ibnetdiscover',
            type=str,
            default=None,
            required=False,
            help='The path of ibnetdiscover output',
        )
        self._parser.add_argument(
            '--direction',
            type=str,
            nargs='+',
            default='gpu-to-gpu',
            required=False,
            help='The direction of traffic pattern, e.g., gpu-to-gpu, cpu-to-cpu, cpu-to-gpu, gpu-to-cpu'
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

    def gen_traffic_pattern(self, hosts, mode, config_file_path):
        """Generate traffic pattern into config file.

        Args:
            hosts (list): the list of VM hostnames read from hostfile.
            mode (str): the traffic mode, including 'one-to-one', 'one-to-many', 'many-to-one', 'topo-aware'.
            config_file_path (str): the path of config file to generate.
        """
        config = []
        n = len(hosts)
        if mode == 'one-to-many':
            config = self.__one_to_many(n)
        elif mode == 'many-to-one':
            config = self.__many_to_one(n)
        elif mode == 'one-to-one':
            config = gen_pair_wise_config(n)
        elif mode == 'topo-aware':
            config = gen_topo_aware_config(
                hosts, self._args.ibstat, self._args.ibnetdiscover, self._args.min_dist, self._args.max_dist
            )
        with open(config_file_path, 'w') as f:
            for line in config:
                f.write(line + '\n')

    def __prepare_config(self):
        """Prepare and read config file.

        Returns:
            True if the config is not empty and valid.
        """
        try:
            # Read the hostfile
            if not self._args.hostfile:
                self._args.hostfile = os.path.join(os.environ.get('SB_WORKSPACE', '.'), 'hostfile')
            with open(self._args.hostfile, 'r') as f:
                hosts = f.read().splitlines()
            # Generate the config file if not define
            if self._args.config is None:
                self.gen_traffic_pattern(hosts, self._args.pattern, self.__config_path)
            # Use the config file defined in args
            else:
                self.__config_path = self._args.config
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

    def __prepare_general_ib_command_params(self, msg_size, device='cpu'):
        """Prepare general params for ib commands.

        Returns:
            Str of ib command params if arguments are valid, otherwise False.
        """
        # Add message size for ib command
        msg_size = f'-s {msg_size}' if msg_size > 0 else '-a'
        # Add GPUDirect for ib command
        gpu_dev = ''
        if device == 'gpu' and self._args.gpu_dev is not None:
            gpu = GPU()
            if gpu.vendor == 'nvidia':
                gpu_dev = f'--use_cuda={self._args.gpu_dev}'
            elif gpu.vendor == 'amd':
                gpu_dev = f'--use_rocm={self._args.gpu_dev}'
            else:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('No GPU found - benchmark: {}'.format(self._name))
                return False
        # Generate ib command params
        command_params = f'-F -n {self._args.iters} -d {self._args.ib_dev} {msg_size} {gpu_dev}'
        command_params = f'{command_params.strip()} --report_gbits'
        return command_params

    def _preprocess(self):    # noqa: C901
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        # Generate and check config
        if not self.__prepare_config():
            return False

        self._commands_ib_commands = []
        self._commands_msg_size = []
        self._commands_direction = []
        if not isinstance(self._args.msg_size, list):
            self._args.msg_size = [self._args.msg_size]
        for msg_size in self._args.msg_size:
            if msg_size < 0:
                logger.error('Invalid message size - benchmark: {}, message size: {}.'.format(self._name, msg_size))
                return False
            # Prepare general params for ib commands
            cpu_command_params = self.__prepare_general_ib_command_params(msg_size)
            gpu_command_params = self.__prepare_general_ib_command_params(msg_size, 'gpu')
            if not cpu_command_params or (self._args.gpu_dev and not gpu_command_params):
                return False
            # Generate commands
            if isinstance(self._args.command, str):
                self._args.command = [self._args.command]
            for ib_command in self._args.command:
                if ib_command not in self.__support_ib_commands:
                    self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                    logger.error(
                        'Unsupported ib command - benchmark: {}, command: {}, expected: {}.'.format(
                            self._name, ib_command, ' '.join(self.__support_ib_commands)
                        )
                    )
                    return False
                else:
                    # Format the ib command type
                    ib_command = ib_command.lower()
                    cpu_ib_command_prefix = f'{os.path.join(self._args.bin_dir, ib_command)} {cpu_command_params}'
                    gpu_ib_command_prefix = f'{os.path.join(self._args.bin_dir, ib_command)} {gpu_command_params}'
                    if self._args.numa_dev is not None:
                        cpu_ib_command_prefix = f'numactl -N {self._args.numa_dev} {cpu_ib_command_prefix}'
                        gpu_ib_command_prefix = f'numactl -N {self._args.numa_dev} {gpu_ib_command_prefix}'
                    if 'bw' in ib_command and self._args.bidirectional:
                        cpu_ib_command_prefix += ' -b'
                        gpu_ib_command_prefix += ' -b'
                    if not isinstance(self._args.direction, list):
                        self._args.direction = [self._args.direction]
                    for direction in self._args.direction:
                        if direction not in self.__support_directions:
                            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                            logger.error(
                                'Unsupported direction - benchmark: {}, direction: {}, expected: {}.'.format(
                                    self._name, direction, ' '.join(self.__support_directions)
                                )
                            )
                            return False
                        # Generate commands
                        command = os.path.join(self._args.bin_dir, self._bin_name)
                        command += ' --send_cmd_prefix ' + "'" + cpu_ib_command_prefix + "'" \
                            if 'cpu-to' in direction else ' --send_cmd_prefix ' + "'" + gpu_ib_command_prefix + "'"
                        command += ' --recv_cmd_prefix ' + "'" + cpu_ib_command_prefix + "'" \
                            if 'to-cpu' in direction else ' --recv_cmd_prefix ' + "'" + gpu_ib_command_prefix + "'"
                        command += f' --timeout {self._args.timeout} ' + \
                            f'--hostfile {self._args.hostfile} --input_config {self.__config_path}'
                        self._commands.append(command)
                        self._commands_ib_commands.append(ib_command)
                        self._commands_msg_size.append(msg_size)
                        self._commands_direction.append(direction)

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
        command = self._commands_ib_commands[cmd_idx]
        msg_size = self._commands_msg_size[cmd_idx]
        direction = self._commands_direction[cmd_idx]
        self._result.add_raw_data(f'raw_output_{command}_{msg_size}_{direction}', raw_output, self._args.log_raw_data)

        # If it's invoked by MPI and rank is not 0, no result is expected
        if os.getenv('OMPI_COMM_WORLD_RANK'):
            rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
            if rank > 0:
                return True

        valid = False
        content = raw_output.splitlines()
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
                for line_index, line in enumerate(content):
                    line_result = list(filter(None, line.strip().split(',')))
                    for pair_index, pair_result in enumerate(line_result):
                        rank_results = list(filter(None, pair_result.strip().split(' ')))
                        for rank_index, rank_result in enumerate(rank_results):
                            metric = f'{command}_{msg_size}_{direction}_{line_index}_{pair_index}:' \
                                + f'{self.__config[config_index]}:{rank_index}'
                            value = float(rank_result)
                            # Check if the value is valid before the base conversion
                            if 'bw' in command and value >= 0.0:
                                value = value / 8.0
                            self._result.add_result(metric, value)
                            valid = True
                        config_index += 1
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
