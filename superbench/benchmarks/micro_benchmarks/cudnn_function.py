# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cudnn functions benchmarks."""

import os
import json
import hashlib
import math
import yaml
import statistics

from superbench.common.utils import logger
from superbench.benchmarks import Platform, BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class CudnnBenchmark(MicroBenchmarkWithInvoke):
    """The cudnn performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self.__default_params_dict_list = [
            {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 0,
                'inputDims': [32, 128, 14, 14],
                'inputStride': [25088, 196, 14, 1],
                'inputType': 0,
                'outputDims': [32, 32, 14, 14],
                'outputStride': [6272, 196, 14, 1],
                'convType': 0,
                'tensorOp': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filterDims': [32, 128, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'inputDims': [32, 128, 14, 14],
                'inputStride': [25088, 196, 14, 1],
                'inputType': 2,
                'outputDims': [32, 32, 14, 14],
                'outputStride': [6272, 196, 14, 1],
                'convType': 0,
                'tensorOp': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filterDims': [32, 128, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'inputDims': [32, 256, 14, 14],
                'inputStride': [50176, 196, 14, 1],
                'inputType': 0,
                'outputDims': [32, 1024, 14, 14],
                'outputStride': [200704, 196, 14, 1],
                'convType': 0,
                'tensorOp': False,
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filterDims': [1024, 256, 1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'inputDims': [32, 256, 14, 14],
                'inputStride': [50176, 196, 14, 1],
                'inputType': 2,
                'outputDims': [32, 1024, 14, 14],
                'outputStride': [200704, 196, 14, 1],
                'convType': 0,
                'tensorOp': True,
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filterDims': [1024, 256, 1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'inputDims': [32, 512, 14, 14],
                'inputStride': [100352, 196, 14, 1],
                'inputType': 0,
                'outputDims': [32, 512, 14, 14],
                'outputStride': [100352, 196, 14, 1],
                'convType': 0,
                'tensorOp': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filterDims': [512, 512, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'inputDims': [32, 512, 14, 14],
                'inputStride': [100352, 196, 14, 1],
                'inputType': 2,
                'outputDims': [32, 512, 14, 14],
                'outputStride': [100352, 196, 14, 1],
                'convType': 0,
                'tensorOp': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filterDims': [512, 512, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filterDims': [32, 128, 3, 3],
                'inputType': 2,
                'inputDims': [32, 32, 14, 14],
                'inputStride': [6272, 196, 14, 1],
                'outputDims': [32, 128, 14, 14],
                'outputStride': [25088, 196, 14, 1],
                'convType': 0,
                'tensorOp': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 4,
                'filterDims': [32, 128, 3, 3],
                'inputType': 0,
                'inputDims': [32, 32, 14, 14],
                'inputStride': [6272, 196, 14, 1],
                'outputDims': [32, 128, 14, 14],
                'outputStride': [25088, 196, 14, 1],
                'convType': 0,
                'tensorOp': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filterDims': [1024, 256, 1, 1],
                'inputType': 0,
                'inputDims': [32, 1024, 14, 14],
                'inputStride': [200704, 196, 14, 1],
                'outputDims': [32, 256, 14, 14],
                'outputStride': [50176, 196, 14, 1],
                'convType': 0,
                'tensorOp': False,
                'arrayLength': 2,
                'padA': [0, 0],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filterDims': [1024, 256, 1, 1],
                'inputType': 2,
                'inputDims': [32, 1024, 14, 14],
                'inputStride': [200704, 196, 14, 1],
                'outputDims': [32, 256, 14, 14],
                'outputStride': [50176, 196, 14, 1],
                'convType': 0,
                'tensorOp': True,
                'arrayLength': 2,
                'padA': [0, 0],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filterDims': [512, 512, 3, 3],
                'inputType': 0,
                'inputDims': [32, 512, 14, 14],
                'inputStride': [100352, 196, 14, 1],
                'outputDims': [32, 512, 14, 14],
                'outputStride': [100352, 196, 14, 1],
                'convType': 0,
                'tensorOp': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filterDims': [512, 512, 3, 3],
                'inputType': 2,
                'inputDims': [32, 512, 14, 14],
                'inputStride': [100352, 196, 14, 1],
                'outputDims': [32, 512, 14, 14],
                'outputStride': [100352, 196, 14, 1],
                'convType': 0,
                'tensorOp': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'inputDims': [32, 128, 14, 14],
                'inputStride': [25088, 196, 14, 1],
                'filterDims': [32, 128, 3, 3],
                'outputDims': [32, 32, 14, 14],
                'outputStride': [6272, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'tensorOp': False,
                'inputType': 0,
                'convType': 0,
                'algo': 6
            }, {
                'name': 'cudnnConvolutionForward',
                'inputDims': [32, 128, 14, 14],
                'inputStride': [25088, 196, 14, 1],
                'filterDims': [32, 128, 3, 3],
                'outputDims': [32, 32, 14, 14],
                'outputStride': [6272, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'tensorOp': True,
                'inputType': 2,
                'convType': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'inputDims': [32, 256, 14, 14],
                'inputStride': [50176, 196, 14, 1],
                'filterDims': [1024, 256, 1, 1],
                'outputDims': [32, 1024, 14, 14],
                'outputStride': [200704, 196, 14, 1],
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'tensorOp': False,
                'inputType': 0,
                'convType': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'inputDims': [32, 256, 14, 14],
                'inputStride': [50176, 196, 14, 1],
                'filterDims': [1024, 256, 1, 1],
                'outputDims': [32, 1024, 14, 14],
                'outputStride': [200704, 196, 14, 1],
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'tensorOp': True,
                'inputType': 2,
                'convType': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'inputDims': [32, 512, 14, 14],
                'inputStride': [100352, 196, 14, 1],
                'filterDims': [512, 512, 3, 3],
                'outputDims': [32, 512, 14, 14],
                'outputStride': [100352, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'tensorOp': False,
                'inputType': 0,
                'convType': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'inputDims': [32, 512, 14, 14],
                'inputStride': [100352, 196, 14, 1],
                'filterDims': [512, 512, 3, 3],
                'outputDims': [32, 512, 14, 14],
                'outputStride': [100352, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'tensorOp': True,
                'inputType': 2,
                'convType': 0,
                'algo': 1
            }
        ]

        self._bin_name = 'cudnn_benchmark'

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
            help='The custom json string defining the params in a cudnn function.',
        )
        self._parser.add_argument(
            '--enable_auto_algo',
            action='store_true',
            default=False,
            required=False,
            help='Whether to use auto algorithm selection.'
        )
        self._parser.add_argument(
            '--execution_mode',
            choices=['legacy', 'prepared'],
            default='legacy',
            help='Prepared mode reuses a cuDNN execution plan for backward-filter only.'
        )
        self._parser.add_argument(
            '--workspace_limit_mib',
            type=int,
            default=1024,
            help='Maximum prepared-plan workspace in MiB.'
        )

    def _execution_config(self, config):
        """Normalize a prepared request without changing legacy configurations."""
        if self._args.execution_mode == 'legacy':
            if 'executionMode' in config:
                raise ValueError('Select prepared execution with --execution_mode prepared.')
            return config
        if self._args.enable_auto_algo:
            raise ValueError('Prepared execution does not use legacy auto algorithm selection.')
        if not 0 <= self._args.workspace_limit_mib <= 1048576:
            raise ValueError('Prepared workspace limit must be between 0 and 1048576 MiB.')
        if (config['name'] != 'cudnnConvolutionBackwardFilter' or config['inputType'] not in (0, 2)
                or config['convType'] != 0):
            raise ValueError('Prepared execution requires backward-filter with FP32 compute and FP32/FP16 storage.')
        config = dict(config)
        config.pop('algo', None)
        config.update(
            executionMode='prepared', planPolicy='deterministic-v1', workspaceLimitMiB=self._args.workspace_limit_mib
        )
        return config

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self._args.tolerant_fail = self._args.execution_mode == 'legacy'
        command = os.path.join(self._args.bin_dir, self._bin_name)
        command += (' --num_test ' + str(self._args.num_steps))
        command += (' --warm_up ' + str(self._args.num_warmup))
        command += (' --num_in_step ' + str(self._args.num_in_step))
        command += (' --random_seed ' + str(self._args.random_seed))
        if self._args.enable_auto_algo:
            command += (' --enable_auto_algo')

        try:
            if not self._args.config_json_str:
                for config_dict in self.__default_params_dict_list:
                    if (self._args.execution_mode == 'prepared'
                            and config_dict['name'] != 'cudnnConvolutionBackwardFilter'):
                        continue
                    config_dict = self._execution_config(config_dict)
                    config_json_str = "\'" + json.dumps(config_dict).replace(' ', '') + "\'"
                    complete_command = command + (' --config_json ') + config_json_str
                    self._commands.append(complete_command)

            else:
                if not isinstance(self._args.config_json_str, list):
                    self._args.config_json_str = [self._args.config_json_str]
                for config_json_str in self._args.config_json_str:
                    custom_config_str = yaml.safe_load(config_json_str)
                    custom_config_str = self._execution_config(custom_config_str)
                    config_json_str = "\'" + json.dumps(custom_config_str).replace(' ', '') + "\'"
                    complete_command = command + (' --config_json ') + config_json_str
                    self._commands.append(complete_command)
        except BaseException as e:
            logger.error('Invalid input params - benchmark: {},  message: {}'.format(self._name, str(e)))
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            return False
        return True

    def _process_prepared_result(self, metric, lines, config):
        """Validate prepared evidence before publishing any timing."""
        metadata_lines = [line for line in lines if line.startswith('[prepared_plan]: ')]
        timing_lines = [line for line in lines if line.startswith('[raw_data]: ')]
        if len(metadata_lines) != 1 or len(timing_lines) != 1 or any('Error' in line for line in lines):
            raise ValueError('Missing, duplicate or failed prepared execution output.')
        metadata = json.loads(metadata_lines[0].split(': ', 1)[1])
        if (metadata['execution_mode'] != 'prepared' or metadata['policy'] != config['planPolicy']
                or not isinstance(metadata['plan'], dict) or not metadata['plan']):
            raise ValueError('Prepared plan identity or policy mismatch.')
        fields = timing_lines[0].split(': ', 1)[1].split(',')
        if fields[-1] != '':
            raise ValueError('Incomplete prepared timing output.')
        raw_data = [float(value) for value in fields[:-1]]
        if len(raw_data) != self._args.num_steps or any(not math.isfinite(value) or value <= 0 for value in raw_data):
            raise ValueError('Invalid prepared timing samples.')
        costs = {name: metadata[name + '_ms'] for name in ('plan_build', 'setup', 'first_call', 'benchmark')}
        if any(type(value) not in (float, int) or not math.isfinite(value) or value < 0 for value in costs.values()):
            raise ValueError('Invalid prepared setup timings.')
        serialized = json.dumps(metadata['plan'], sort_keys=True, separators=(',', ':')).encode()
        metric = metric.lower() + '_plan_' + hashlib.sha256(serialized).hexdigest()[:16]
        self._result.add_result(metric + '_time', statistics.mean(raw_data) * 1000)
        self._result.add_raw_data(metric + '_time', raw_data, self._args.log_raw_data)
        for name, value in costs.items():
            self._result.add_result(metric + '_' + name + '_time', value * 1000)
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
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output, self._args.log_raw_data)
        metric = ''
        try:
            lines = raw_output.splitlines()

            cmd_config = json.loads(self._commands[cmd_idx].split('--config_json')[-1].replace(' ', '')[1:-1])
            for key in sorted(cmd_config.keys()):
                if 'name' in key:
                    metric = key + '_' + str(cmd_config[key]) + metric
                else:
                    metric = metric + '_' + key + '_' + str(cmd_config[key])
            metric = metric.replace(' ', '').replace(',', '_')

            if cmd_config.get('executionMode') == 'prepared':
                return self._process_prepared_result(metric, lines, cmd_config)

            error = False
            raw_data = []
            for line in lines:
                if '[raw_data]' in line:
                    raw_data = line[line.index('[raw_data]: ') + len('[raw_data]: '):]
                    raw_data = raw_data.split(',')
                    raw_data.pop()
                    raw_data = [float(item) for item in raw_data]
                    self._result.add_result(metric.lower() + '_time', statistics.mean(raw_data) * 1000)
                    self._result.add_raw_data(metric.lower() + '_time', raw_data, self._args.log_raw_data)
                if 'Error' in line:
                    error = True
        except BaseException as e:
            logger.error(
                'Cannot extract results from cudnn functions - round: {}, index of cmd: {}, \
                benchmark: {}, raw data: {}, message: {}'.format(
                    self._curr_run_index, cmd_idx, self._name, raw_output, str(e)
                )
            )
            error = True
        if error:
            logger.error(
                'Error in running cudnn test - round: {}, index of cmd: {}, benchmark: {}, raw data: {}'.format(
                    self._curr_run_index, cmd_idx, self._name, raw_output
                )
            )
            self._result.add_result(metric.lower() + '_time', -1)
            return False
        return True


BenchmarkRegistry.register_benchmark('cudnn-function', CudnnBenchmark, platform=Platform.CUDA)
