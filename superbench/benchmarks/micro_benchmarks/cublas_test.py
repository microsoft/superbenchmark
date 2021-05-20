# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cublas functions benchmarks."""

import os

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CublasFunction(MicroBenchmarkWithInvoke):
    """The CublasFunction overhead benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'CublasBenchmark'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=8,
            required=False,
            help='The number of warmup step.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=100,
            required=False,
            help='The number of test step.',
        )
        self._parser.add_argument(
            '--num_in_step',
            type=int,
            default=1000,
            required=False,
            help='The number of functions in one step.',
        )
        self._parser.add_argument(
            '--config_path',
            type=str,
            default='/opt/superbench/superbench/benchmarks/micro_benchmarks/cublas_function/para_info.json',
            required=False,
            help='The path of functions config json file.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        command = os.path.join(self._args.bin_dir, self._bin_name)
        command += (' --num_test ' + str(self._args.num_steps))
        command += (' --warm_up ' + str(self._args.num_warmup))
        command += (' --num_in_step ' + str(self._args.num_in_step))
        command += (' --config_path ' + str(self._args.config_path))
        self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output)

        try:
            lines = raw_output.split('\n')
            metric = ''
            raw_data = []
            for line in lines:
                if '[function config]' in line:
                    metric = line[line.index('[function config]: ') + len('[function config]: '):]
                if '[raw_data]' in line:
                    raw_data = line[line.index('[raw_data]: ') + len('[raw_data]: '):]
                    raw_data = raw_data.split(',')
                    raw_data.pop()
                    raw_data = [float(item) for item in raw_data]
                    self._result.add_result(metric, sum(raw_data) / len(raw_data))
                    self._result.add_raw_data(metric, raw_data)
                if 'Error' in line:
                    raise Exception(line)
        except BaseException as e:
            logger.error(
                'Cannot extract results from cublas functions - round: {}, benchmark: {}, raw data: {}, message: {}'.
                format(self._curr_run_index, self._name, raw_output, str(e))
            )
            return False
        return True


BenchmarkRegistry.register_benchmark('cublas-test', CublasFunction)
