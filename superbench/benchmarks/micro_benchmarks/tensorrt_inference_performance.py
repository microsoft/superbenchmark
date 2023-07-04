# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TensorRT inference micro-benchmark."""

import re
from pathlib import Path

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke
from superbench.benchmarks.micro_benchmarks._export_torch_to_onnx import torch2onnxExporter


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
        self._pytorch_models = ['resnet50']

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
            choices=['int8', 'fp16', 'fp32'],
            default='int8',
            required=False,
            help='Precision for inference, allow int8, fp16, or fp32 only.',
        )

        self._parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            required=False,
            help='Set batch size for inference input.',
        )

        self._parser.add_argument(
            '--seq_length',
            type=int,
            default=512,
            required=False,
            help='Set sequence length for inference input, only effective for transformers',
        )

        self._parser.add_argument(
            '--iterations',
            type=int,
            default=2048,
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

        exporter = torch2onnxExporter()
        for model in self._args.pytorch_models:
            if not (exporter.check_torchvision_model(model) or exporter.check_benchmark_model(model)):
                logger.error('Cannot find PyTorch model %s.', model)
                return False
        for model in self._args.pytorch_models:
            input_shape: str
            onnx_model: str
            if exporter.check_torchvision_model(model):
                input_shape = f'{self._args.batch_size}x3x224x224'
                onnx_model = exporter.export_torchvision_model(model, self._args.batch_size)
            if exporter.check_benchmark_model(model):
                input_shape = f'{self._args.batch_size}x{self._args.seq_length}'
                onnx_model = exporter.export_benchmark_model(model, self._args.batch_size, self._args.seq_length)
            args = [
                # trtexec
                self.__bin_path,
                # model options
                f'--onnx={onnx_model}',
                # build options
                '--explicitBatch',
                f'--optShapes=input:{input_shape}',
                '--workspace=8192',
                None if self._args.precision == 'fp32' else f'--{self._args.precision}',
                # inference options
                f'--iterations={self._args.iterations}',
                # reporting options
                '--percentile=99',
            ]   # yapf: disable
            self._commands.append(' '.join(filter(None, args)))

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
            f'raw_output_{self._args.pytorch_models[cmd_idx]}', raw_output, self._args.log_raw_data
        )

        success = False
        try:
            model = self._args.pytorch_models[cmd_idx]
            for line in raw_output.strip().splitlines():
                line = line.strip()
                if '[I] mean:' in line or '[I] percentile:' in line:
                    tag = 'mean' if '[I] mean:' in line else '99'
                    lats = re.findall(r'(\d+\.\d+) ms', line)
                    if len(lats) == 1:
                        self._result.add_result(f'{model}_gpu_time_{tag}', float(lats[0]))
                    elif len(lats) == 2:
                        self._result.add_result(f'{model}_host_time_{tag}', float(lats[0]))
                        self._result.add_result(f'{model}_end_to_end_time_{tag}', float(lats[1]))
                    success = True
                if '[I] Latency:' in line or '[I] GPU Compute Time:' in line:
                    tm = 'gpu' if '[I] GPU Compute Time:' in line else 'host'
                    self._result.add_result(
                        f'{model}_{tm}_time_mean',
                        float(re.findall(r'mean = (\d+\.\d+) ms', line)[0]),
                    )
                    self._result.add_result(
                        f'{model}_{tm}_time_99',
                        float(re.findall(r'\(99\%\) = (\d+\.\d+) ms', line)[0]),
                    )
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
