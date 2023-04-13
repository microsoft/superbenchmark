# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cublas functions benchmarks."""

import os
import json
import yaml
import statistics

from superbench.common.utils import logger
from superbench.benchmarks import Platform, BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CublasBenchmark(MicroBenchmarkWithInvoke):
    """The Cublas performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self.__default_params_dict_list = [
            {
                'name': 'cublasCgemm',
                'm': 512,
                'n': 512,
                'k': 32,
                'transa': 1,
                'transb': 0
            }, {
                'name': 'cublasCgemm',
                'm': 2048,
                'n': 512,
                'k': 32,
                'transa': 1,
                'transb': 0
            }, {
                'name': 'cublasCgemm',
                'm': 512,
                'n': 2048,
                'k': 32,
                'transa': 1,
                'transb': 0
            }, {
                'name': 'cublasCgemm',
                'm': 640,
                'n': 1280,
                'k': 32,
                'transa': 1,
                'transb': 0
            }, {
                'name': 'cublasCgemm',
                'm': 896,
                'n': 1792,
                'k': 32,
                'transa': 1,
                'transb': 0
            }, {
                'name': 'cublasCgemm3mStridedBatched',
                'm': 64,
                'n': 32,
                'k': 3,
                'transa': 0,
                'transb': 1,
                'batchCount': 544
            }, {
                'name': 'cublasCgemm3mStridedBatched',
                'm': 64,
                'n': 32,
                'k': 64,
                'transa': 1,
                'transb': 0,
                'batchCount': 544
            }, {
                'name': 'cublasCgemm3mStridedBatched',
                'm': 128,
                'n': 32,
                'k': 128,
                'transa': 0,
                'transb': 1,
                'batchCount': 544
            }, {
                'name': 'cublasCgemm3mStridedBatched',
                'm': 128,
                'n': 32,
                'k': 64,
                'transa': 0,
                'transb': 1,
                'batchCount': 544
            }, {
                'name': 'cublasCgemm3mStridedBatched',
                'm': 64,
                'n': 32,
                'k': 128,
                'transa': 0,
                'transb': 1,
                'batchCount': 544
            }, {
                'name': 'cublasGemmStridedBatchedEx',
                'm': 224,
                'n': 224,
                'k': 64,
                'transa': 0,
                'transb': 0,
                'datatype': 'half',
                'use_tensor_core': True,
                'batchCount': 160
            }, {
                'name': 'cublasGemmStridedBatchedEx',
                'm': 64,
                'n': 224,
                'k': 224,
                'transa': 0,
                'transb': 0,
                'datatype': 'half',
                'use_tensor_core': True,
                'batchCount': 160
            }, {
                'name': 'cublasGemmEx',
                'm': 4000,
                'n': 224,
                'k': 1000,
                'transa': 0,
                'transb': 0,
                'datatype': 'float',
                'use_tensor_core': False
            }, {
                'name': 'cublasGemmEx',
                'm': 4000,
                'n': 224,
                'k': 1000,
                'transa': 1,
                'transb': 0,
                'datatype': 'half',
                'use_tensor_core': True
            }, {
                'name': 'cublasGemmEx',
                'm': 1000,
                'n': 224,
                'k': 4000,
                'transa': 0,
                'transb': 0,
                'datatype': 'half',
                'use_tensor_core': False
            }, {
                'name': 'cublasGemmEx',
                'm': 1000,
                'n': 224,
                'k': 4000,
                'transa': 0,
                'transb': 0,
                'datatype': 'float',
                'use_tensor_core': False
            }, {
                'name': 'cublasSgemm',
                'm': 1024,
                'n': 7168,
                'k': 1024,
                'transa': 1,
                'transb': 0
            }, {
                'name': 'cublasSgemmStridedBatched',
                'm': 64,
                'n': 224,
                'k': 224,
                'transa': 0,
                'transb': 0,
                'batchCount': 512
            }, {
                'name': 'cublasSgemmStridedBatched',
                'm': 64,
                'n': 224,
                'k': 224,
                'transa': 0,
                'transb': 0,
                'batchCount': 160
            }
        ]

        self._bin_name = 'cublas_benchmark'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=8 * 1000,
            required=False,
            help='The number of functions for warmup. By default, the total number of functions to run in warmup ' +
            'is 8 warmup steps * 1000 num_in_step.',
        )
        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=100,
            required=False,
            help='The number of test steps. By default, the total number of functions to run in the measured test ' +
            'is 100 test steps * 1000 num_in_step.',
        )
        self._parser.add_argument(
            '--num_in_step',
            type=int,
            default=1000,
            required=False,
            help='The number of functions in one step. By default, the total number of functions to run ' +
            'in each step is 1000.',
        )
        self._parser.add_argument(
            '--random_seed',
            type=int,
            default=33931,
            required=False,
            help='The random seed to fill in the data of the function.',
        )
        self._parser.add_argument(
            '--config_json_str',
            type=str,
            nargs='+',
            default=None,
            required=False,
            help='The custom json string defining the params in a cublas function.',
        )
        self._parser.add_argument(
            '--correctness',
            action='store_true',
            default=False,
            help='Enable correctness check for cublas functions.',
        )
        self._parser.add_argument(
            '--eps',
            type=float,
            default=None,
            required=False,
            help='The acceptable error bound for correctness check.',
        )
        self._parser.add_argument(
            '--random_data',
            action='store_true',
            default=False,
            help='Enable random data generation for performance test. ' +
            'By default, the data is filled with fixed value for performance test.',
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
        command += (' --random_seed ' + str(self._args.random_seed))
        command += ' --correctness' if self._args.correctness else ''
        command += (' --eps ' + str(self._args.eps)) if self._args.eps is not None else ''
        command += ' --random_data' if self._args.random_data else ''

        try:
            if not self._args.config_json_str:
                for config_dict in self.__default_params_dict_list:
                    config_json_str = "\'" + json.dumps(config_dict).replace(' ', '') + "\'"
                    complete_command = command + (' --config_json ') + config_json_str
                    self._commands.append(complete_command)

            else:
                if not isinstance(self._args.config_json_str, list):
                    self._args.config_json_str = [self._args.config_json_str]
                for config_json_str in self._args.config_json_str:
                    custom_config_str = yaml.safe_load(config_json_str)
                    config_json_str = "\'" + json.dumps(custom_config_str).replace(' ', '') + "\'"
                    complete_command = command + (' --config_json ') + config_json_str
                    self._commands.append(complete_command)
        except BaseException as e:
            logger.error('Invalid input params - benchmark: {},  message: {}'.format(self._name, str(e)))
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False
        return True

    def _process_raw_result(self, cmd_idx, raw_output):    # noqa: C901
        """Function to process raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)

        try:
            lines = raw_output.splitlines()
            metric = ''
            error = False
            raw_data = []
            for line in lines:
                if '[function config]' in line:
                    metric = ''
                    metric_json_str = line[line.index('[function config]: ') +
                                           len('[function config]: '):].replace(' ', '').replace(':', '_')[1:-1]
                    metric_list = metric_json_str.split(',')
                    for key in metric_list:
                        if 'name' in key:
                            metric = key + metric
                        else:
                            metric = metric + '_' + key
                if '[raw_data]' in line:
                    raw_data = line[line.index('[raw_data]: ') + len('[raw_data]: '):]
                    raw_data = raw_data.split(',')
                    raw_data.pop()
                    raw_data = [float(item) for item in raw_data]
                    self._result.add_result(metric.lower() + '_time', statistics.mean(raw_data))
                    self._result.add_raw_data(metric.lower() + '_time', raw_data, self._args.log_raw_data)
                if 'Error' in line:
                    error = True
                if '[correctness]' in line:
                    if 'PASS' in line:
                        self._result.add_result(metric.lower() + '_correctness', 1)
                    elif 'FAIL' in line:
                        self._result.add_result(metric.lower() + '_correctness', 0)
                    error_rate = float(line.split(' ')[-1])
                    self._result.add_result(metric.lower() + '_error_rate', error_rate)

        except BaseException as e:
            logger.error(
                'Cannot extract results from cublas functions - round: {}, index of cmd: {}, \
                benchmark: {}, raw data: {}, message: {}'.format(
                    self._curr_run_index, cmd_idx, self._name, raw_output, str(e)
                )
            )
            return False
        if error:
            logger.error(
                'Error in running cublas test - round: {}, index of cmd: {}, benchmark: {}, raw data: {}'.format(
                    self._curr_run_index, cmd_idx, self._name, raw_output
                )
            )
            return False
        return True


BenchmarkRegistry.register_benchmark('cublas-function', CublasBenchmark, platform=Platform.CUDA)
