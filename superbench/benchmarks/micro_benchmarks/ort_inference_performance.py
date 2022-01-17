# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TensorRT inference micro-benchmark."""

import time
import statistics
from pathlib import Path

import torch
import torchvision.models
import numpy as np

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, Precision
from superbench.benchmarks.micro_benchmarks import MicroBenchmark


class ORTInferenceBenchmark(MicroBenchmark):
    """ONNXRuntime inference micro-benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

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
        self.__graph_opt_level = None
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
            type=Precision,
            choices=[Precision.FLOAT32, Precision.FLOAT16, Precision.INT8],
            default=Precision.FLOAT16,
            required=False,
            help='Precision for inference, allow int8, float16, or float32 only.',
        )

        self._parser.add_argument(
            '--graph_opt_level',
            type=int,
            default=3,
            choices=[0, 1, 2, 3],
            required=False,
            help='ONNXRuntime graph optimization level, 0 for ORT_DISABLE_ALL, 1 for ORT_ENABLE_BASIC, '
            '2 for ORT_ENABLE_EXTENDED, 3 for ORT_ENABLE_ALL.',
        )

        self._parser.add_argument(
            '--batch_size',
            type=int,
            default=32,
            required=False,
            help='Set batch size for inference.',
        )

        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=64,
            required=False,
            help='The number of warmup step before the benchmarking.',
        )

        self._parser.add_argument(
            '--num_steps',
            type=int,
            default=256,
            required=False,
            help='The number of test step for benchmarking.',
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        import onnxruntime as ort
        self.__graph_opt_level = {
            0: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            1: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            2: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            3: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }

        for model in self._args.pytorch_models:
            if hasattr(torchvision.models, model):
                data_type = Precision.FLOAT16.value if self._args.precision == Precision.FLOAT16 \
                    else Precision.FLOAT32.value
                model_path = f'{self.__model_cache_path / (model + "." + data_type + ".onnx")}'
                torch.onnx.export(
                    getattr(torchvision.models, model)(pretrained=True).to(dtype=getattr(torch, data_type)).cuda(),
                    torch.randn(self._args.batch_size, 3, 224, 224, device='cuda', dtype=getattr(torch, data_type)),
                    model_path,
                    input_names=['input'],
                )
                if self._args.precision == Precision.INT8:
                    file_name = '{model}.{precision}.onnx'.format(model=model, precision=self._args.precision)
                    # For quantization of ONNXRuntime, refer
                    # https://onnxruntime.ai/docs/performance/quantization.html#quantization-overview
                    from onnxruntime.quantization import quantize_dynamic
                    quantize_dynamic(model_path, f'{self.__model_cache_path / file_name}')
            else:
                logger.error('Cannot find PyTorch model %s.', model)
                return False

        return True

    def _benchmark(self):
        """Implementation for benchmarking."""
        import onnxruntime as ort
        precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'int8': 'int8'}

        for model in self._args.pytorch_models:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = self.__graph_opt_level[self._args.graph_opt_level]
            file_name = '{model}.{precision}.onnx'.format(model=model, precision=self._args.precision)
            ort_sess = ort.InferenceSession(
                f'{self.__model_cache_path / file_name}', sess_options, providers=['CUDAExecutionProvider']
            )

            elapse_times = self.__inference(ort_sess)

            if self._args.precision.value in precision_metric:
                precision = precision_metric[self._args.precision.value]
            else:
                precision = self._args.precision.value
            metric = '{}_{}_time'.format(precision, model)
            if not self._process_numeric_result(metric, elapse_times, cal_percentile=True):
                return False

            logger.info(
                'ORT Inference - round: {}, name: {}, model: {}, precision: {}, latency: {} ms'.format(
                    self._curr_run_index, self._name, model, self._args.precision, statistics.mean(elapse_times)
                )
            )

        return True

    def __inference(self, ort_sess):
        """Do inference given the ORT inference session.

        Args:
            ort_sess (InferenceSession): inference session for ORT.

        Return:
            elapse_times (List[float]): latency of every iterations.
        """
        precision = np.float16 if self._args.precision == Precision.FLOAT16 else np.float32
        input_tensor = np.random.randn(self._args.batch_size, 3, 224, 224).astype(dtype=precision)

        for i in range(self._args.num_warmup):
            ort_sess.run(None, {'input': input_tensor})

        elapse_times = list()
        for i in range(self._args.num_steps):
            start = time.time()
            ort_sess.run(None, {'input': input_tensor})
            end = time.time()
            elapse_times.append((end - start) * 1000)

        return elapse_times


BenchmarkRegistry.register_benchmark(
    'ort-inference',
    ORTInferenceBenchmark,
    platform=Platform.CUDA,
)
