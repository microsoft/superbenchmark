# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cudnn functions benchmarks."""

import os
import json
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

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self._args.tolerant_fail = True
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
