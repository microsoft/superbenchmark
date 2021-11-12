# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TensorRT inference micro-benchmark."""

import re
from pathlib import Path

import torch.hub
import torch.onnx
import torchvision.models

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


class TensorRTInferenceBenchmark(MicroBenchmarkWithInvoke):
    """TensorRT inference micro-benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'trtexec'
        self._pytorch_models = [
            'resnet50',
            'resnet101',
            'resnet152',
            'densenet169',
            'densenet201',
            'vgg11',
            'vgg13',
            'vgg16',
            'vgg19',
        ]
        self.__model_cache_path = Path(torch.hub.get_dir()) / 'checkpoints'

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--pytorch_models',
            type=str,
            nargs='+',
            default=self._pytorch_models,
            help='ONNX models for TensorRT inference benchmark, e.g., {}.'.format(', '.join(self._pytorch_models)),
        )

        self._parser.add_argument(
            '--precision',
            type=str,
            choices=['int8', 'fp16'],
            default='int8',
            required=False,
            help='Precision for inference, allowe int8 or fp16.',
        )

        self._parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            required=False,
            help='Set batch size for implicit batch engines.',
        )

        self._parser.add_argument(
            '--iterations',
            type=int,
            default=256,
            required=False,
            help='Run at least N inference iterations.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = str(Path(self._args.bin_dir) / self._bin_name)

        for model in self._args.pytorch_models:
            if hasattr(torchvision.models, model):
                torch.onnx.export(
                    getattr(torchvision.models, model)(pretrained=True).cuda(),
                    torch.randn(self._args.batch_size, 3, 224, 224, device='cuda'),
                    f'{self.__model_cache_path / (model + ".onnx")}',
                )
                self._commands.append(
                    ' '.join(
                        [
                            self.__bin_path,
                            f'--{self._args.precision}',
                            f'--batch={self._args.batch_size}',
                            f'--iterations={self._args.iterations}',
                            '--workspace=1024',
                            '--percentile=99',
                            f'--onnx={self.__model_cache_path / (model + ".onnx")}',
                        ]
                    )
                )
            else:
                logger.error('Cannot find PyTorch model %s.', model)
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
        self._result.add_raw_data(f'raw_output_{self._args.pytorch_models[cmd_idx]}', raw_output)

        success = False
        try:
            for line in raw_output.strip().splitlines():
                line = line.strip()
                if '[I] mean:' in line or '[I] percentile:' in line:
                    tag = 'mean' if '[I] mean:' in line else '99'
                    lats = re.findall(r'(\d+\.\d+) ms', line)
                    if len(lats) == 1:
                        self._result.add_result(f'gpu_lat_ms_{tag}', float(lats[0]))
                    elif len(lats) == 2:
                        self._result.add_result(f'host_lat_ms_{tag}', float(lats[0]))
                        self._result.add_result(f'end_to_end_lat_ms_{tag}', float(lats[1]))
                    success = True
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}, message: {}.'.format(
                    self._curr_run_index, self._name, raw_output, str(e)
                )
            )
            return False
        return success


BenchmarkRegistry.register_benchmark(
    'tensorrt-inference',
    TensorRTInferenceBenchmark,
    platform=Platform.CUDA,
)
