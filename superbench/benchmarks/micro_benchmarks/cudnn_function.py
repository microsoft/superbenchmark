# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the cudnn functions benchmarks."""

import os
import json
import yaml

from superbench.common.utils import logger
from superbench.benchmarks import Platform, BenchmarkRegistry
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
                'input_dims': [32, 128, 14, 14],
                'input_stride': [25088, 196, 14, 1],
                'input_type': 0,
                'output_dims': [32, 32, 14, 14],
                'output_stride': [6272, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filter_dims': [32, 128, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'input_dims': [32, 128, 14, 14],
                'input_stride': [25088, 196, 14, 1],
                'input_type': 2,
                'output_dims': [32, 32, 14, 14],
                'output_stride': [6272, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filter_dims': [32, 128, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'input_dims': [32, 256, 14, 14],
                'input_stride': [50176, 196, 14, 1],
                'input_type': 0,
                'output_dims': [32, 1024, 14, 14],
                'output_stride': [200704, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': False,
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filter_dims': [1024, 256, 1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'input_dims': [32, 256, 14, 14],
                'input_stride': [50176, 196, 14, 1],
                'input_type': 2,
                'output_dims': [32, 1024, 14, 14],
                'output_stride': [200704, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': True,
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filter_dims': [1024, 256, 1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'input_dims': [32, 512, 14, 14],
                'input_stride': [100352, 196, 14, 1],
                'input_type': 0,
                'output_dims': [32, 512, 14, 14],
                'output_stride': [100352, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filter_dims': [512, 512, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardFilter',
                'algo': 1,
                'input_dims': [32, 512, 14, 14],
                'input_stride': [100352, 196, 14, 1],
                'input_type': 2,
                'output_dims': [32, 512, 14, 14],
                'output_stride': [100352, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'filter_dims': [512, 512, 3, 3],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filter_dims': [32, 128, 3, 3],
                'input_type': 2,
                'input_dims': [32, 32, 14, 14],
                'input_stride': [6272, 196, 14, 1],
                'output_dims': [32, 128, 14, 14],
                'output_stride': [25088, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 4,
                'filter_dims': [32, 128, 3, 3],
                'input_type': 0,
                'input_dims': [32, 32, 14, 14],
                'input_stride': [6272, 196, 14, 1],
                'output_dims': [32, 128, 14, 14],
                'output_stride': [25088, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filter_dims': [1024, 256, 1, 1],
                'input_type': 0,
                'input_dims': [32, 1024, 14, 14],
                'input_stride': [200704, 196, 14, 1],
                'output_dims': [32, 256, 14, 14],
                'output_stride': [50176, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': False,
                'arrayLength': 2,
                'padA': [0, 0],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filter_dims': [1024, 256, 1, 1],
                'input_type': 2,
                'input_dims': [32, 1024, 14, 14],
                'input_stride': [200704, 196, 14, 1],
                'output_dims': [32, 256, 14, 14],
                'output_stride': [50176, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': True,
                'arrayLength': 2,
                'padA': [0, 0],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filter_dims': [512, 512, 3, 3],
                'input_type': 0,
                'input_dims': [32, 512, 14, 14],
                'input_stride': [100352, 196, 14, 1],
                'output_dims': [32, 512, 14, 14],
                'output_stride': [100352, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': False,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionBackwardData',
                'algo': 1,
                'filter_dims': [512, 512, 3, 3],
                'input_type': 2,
                'input_dims': [32, 512, 14, 14],
                'input_stride': [100352, 196, 14, 1],
                'output_dims': [32, 512, 14, 14],
                'output_stride': [100352, 196, 14, 1],
                'conv_type': 0,
                'use_tensor_core': True,
                'arrayLength': 2,
                'padA': [1, 1],
                'dilationA': [1, 1],
                'filterStrideA': [1, 1],
                'mode': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'input_dims': [32, 128, 14, 14],
                'input_stride': [25088, 196, 14, 1],
                'filter_dims': [32, 128, 3, 3],
                'output_dims': [32, 32, 14, 14],
                'output_stride': [6272, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'use_tensor_core': False,
                'input_type': 0,
                'conv_type': 0,
                'algo': 6
            }, {
                'name': 'cudnnConvolutionForward',
                'input_dims': [32, 128, 14, 14],
                'input_stride': [25088, 196, 14, 1],
                'filter_dims': [32, 128, 3, 3],
                'output_dims': [32, 32, 14, 14],
                'output_stride': [6272, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'use_tensor_core': True,
                'input_type': 2,
                'conv_type': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'input_dims': [32, 256, 14, 14],
                'input_stride': [50176, 196, 14, 1],
                'filter_dims': [1024, 256, 1, 1],
                'output_dims': [32, 1024, 14, 14],
                'output_stride': [200704, 196, 14, 1],
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'use_tensor_core': False,
                'input_type': 0,
                'conv_type': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'input_dims': [32, 256, 14, 14],
                'input_stride': [50176, 196, 14, 1],
                'filter_dims': [1024, 256, 1, 1],
                'output_dims': [32, 1024, 14, 14],
                'output_stride': [200704, 196, 14, 1],
                'arrayLength': 2,
                'padA': [0, 0],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'use_tensor_core': True,
                'input_type': 2,
                'conv_type': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'input_dims': [32, 512, 14, 14],
                'input_stride': [100352, 196, 14, 1],
                'filter_dims': [512, 512, 3, 3],
                'output_dims': [32, 512, 14, 14],
                'output_stride': [100352, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'use_tensor_core': False,
                'input_type': 0,
                'conv_type': 0,
                'algo': 1
            }, {
                'name': 'cudnnConvolutionForward',
                'input_dims': [32, 512, 14, 14],
                'input_stride': [100352, 196, 14, 1],
                'filter_dims': [512, 512, 3, 3],
                'output_dims': [32, 512, 14, 14],
                'output_stride': [100352, 196, 14, 1],
                'arrayLength': 2,
                'padA': [1, 1],
                'filterStrideA': [1, 1],
                'dilationA': [1, 1],
                'mode': 1,
                'use_tensor_core': True,
                'input_type': 2,
                'conv_type': 0,
                'algo': 1
            }
        ]

        self._bin_name = 'CudnnBenchmark'

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
            default=None,
            required=False,
            help='The custom json string defining the params in a cudnn function.',
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

        try:
            if not self._args.config_json_str:
                for config_dict in self.__default_params_dict_list:
                    config_json_str = "\'" + json.dumps(config_dict).replace(' ', '') + "\'"
                    print(config_json_str)
                    complete_command = command + (' --config_json ') + config_json_str
                    self._commands.append(complete_command)

            else:
                custom_config_str = yaml.safe_load(self._args.config_json_str)
                config_json_str = "\'" + json.dumps(custom_config_str).replace(' ', '') + "\'"
                complete_command = command + (' --config_json ') + config_json_str
                self._commands.append(complete_command)
        except BaseException as e:
            logger.error('Invalid input params - benchmark: {},  message: {}'.format(self._name, str(e)))
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
        self._result.add_raw_data('raw_output_' + str(cmd_idx), raw_output)

        try:
            lines = raw_output.splitlines()
            metric = ''
            error = False
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
                    error = True
        except BaseException as e:
            logger.error(
                'Cannot extract results from cudnn functions - round: {}, index of cmd: {}, \
                benchmark: {}, raw data: {}, message: {}'.format(
                    self._curr_run_index, cmd_idx, self._name, raw_output, str(e)
                )
            )
            return False
        if error:
            logger.error(
                'Error in running cudnn test - round: {}, index of cmd: {}, benchmark: {}, raw data: {}'.format(
                    self._curr_run_index, cmd_idx, self._name, raw_output
                )
            )
            return False
        return True


BenchmarkRegistry.register_benchmark('cudnn-function', CudnnBenchmark, platform=Platform.CUDA)
