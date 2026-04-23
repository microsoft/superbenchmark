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
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import HuggingFaceModelLoader


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

        # HuggingFace model arguments
        self._parser.add_argument(
            '--model_source',
            type=str,
            choices=['in-house', 'huggingface'],
            default='in-house',
            required=False,
            help='Source of the model: inhouse (default) or huggingface.',
        )

        self._parser.add_argument(
            '--model_identifier',
            type=str,
            default=None,
            required=False,
            help='Model identifier for HuggingFace models (e.g., bert-base-uncased).',
        )

        self._parser.add_argument(
            '--seq_length',
            type=int,
            default=512,
            required=False,
            help='Sequence length for transformer models.',
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

        # Handle HuggingFace models if specified
        if self._args.model_source == 'huggingface':
            return self._preprocess_huggingface_models()

        # Original in-house model processing
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

    def _preprocess_huggingface_models(self):
        """Preprocess HuggingFace models for ONNX Runtime inference.

        Returns:
            bool: True if preprocessing succeeds.
        """
        import os

        if not self._args.model_identifier:
            logger.error('--model_identifier is required when using --model_source huggingface')
            return False

        try:
            logger.info(f'Loading HuggingFace model: {self._args.model_identifier}')

            # Step 1: Pre-download memory check — download config only (few KB)
            from transformers import AutoConfig
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            load_kwargs = {}
            if hf_token:
                load_kwargs['token'] = hf_token
            hf_config = AutoConfig.from_pretrained(self._args.model_identifier, trust_remote_code=True, **load_kwargs)

            precision_str = self._args.precision.value if self._args.precision != Precision.INT8 else 'float32'
            fits, param_m, est_gb, avail_gb = HuggingFaceModelLoader.check_memory_fits(
                self._args.model_identifier, hf_config, precision_str, mode='inference', token=hf_token
            )
            if not fits:
                return False

            # Step 2: Proceed with model download and ONNX export

            # Get GPU rank to create unique file paths and avoid race conditions
            # when multiple processes export the same model simultaneously
            gpu_rank = os.getenv('CUDA_VISIBLE_DEVICES', '0')
            proc_rank = os.getenv('PROC_RANK', gpu_rank)

            # Create model source config - load on CPU to avoid accelerate dispatching
            # model across multiple GPUs which causes device mismatch during ONNX export
            model_config = ModelSourceConfig(
                source='huggingface',
                identifier=self._args.model_identifier,
                hf_token=hf_token,
                torch_dtype=self._args.precision.value if self._args.precision != Precision.INT8 else 'float32',
                device_map=None,
            )

            # Load model from HuggingFace on CPU
            loader = HuggingFaceModelLoader()
            hf_model, _, _ = loader.load_model_from_config(model_config, device='cpu')
            from superbench.benchmarks.micro_benchmarks._export_torch_to_onnx import torch2onnxExporter
            exporter = torch2onnxExporter()

            model_name = self._args.model_identifier.replace('/', '_')

            # Prepare output path - use proc_rank subdirectory to avoid race conditions
            # when multiple processes export the same model simultaneously
            proc_output_path = self.__model_cache_path / f'rank_{proc_rank}'
            proc_output_path.mkdir(parents=True, exist_ok=True)

            # For INT8, export as float32 first then quantize (matching in-house model behavior).
            # For other precisions, include precision in the model name directly.
            if self._args.precision == Precision.INT8:
                export_precision = Precision.FLOAT32.value
            else:
                export_precision = self._args.precision.value
            model_name_with_precision = f'{model_name}.{export_precision}'

            # Export directly to final destination to avoid path issues with external data
            onnx_path = exporter.export_huggingface_model(
                model=hf_model,
                model_name=model_name_with_precision,
                batch_size=self._args.batch_size,
                seq_length=self._args.seq_length,
                output_dir=str(proc_output_path),
            )

            if not onnx_path:
                logger.error(f'Failed to export {self._args.model_identifier} to ONNX')
                return False

            # Apply INT8 quantization if requested (matching in-house model behavior)
            if self._args.precision == Precision.INT8:
                from onnxruntime.quantization import quantize_dynamic
                quantized_path = str(proc_output_path / f'{model_name}.{Precision.INT8.value}.onnx')
                quantize_dynamic(onnx_path, quantized_path)
                logger.info('Applied INT8 quantization to HuggingFace model')

            # Update model list and cache path for benchmarking
            self._args.pytorch_models = [model_name]
            self.__model_cache_path = proc_output_path

            logger.info('Successfully prepared HuggingFace model for ORT inference')
            return True

        except Exception as e:
            logger.error(f'Failed to prepare HuggingFace model: {str(e)}')
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _benchmark(self):
        """Implementation for benchmarking."""
        import onnxruntime as ort
        precision_metric = {'float16': 'fp16', 'float32': 'fp32', 'int8': 'int8'}

        # Require CUDAExecutionProvider — this benchmark targets GPU inference
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in available:
            logger.error(f'CUDAExecutionProvider is not available (available: {available}).')
            return False

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

        # Get input names from the ONNX session to determine input format
        input_names = [input.name for input in ort_sess.get_inputs()]

        # Determine input format based on what the model expects
        if 'pixel_values' in input_names:
            # Vision model: use pixel_values (batch_size, 3, 224, 224)
            pixel_values = np.random.randn(self._args.batch_size, 3, 224, 224).astype(dtype=precision)
            inputs = {'pixel_values': pixel_values}
        elif 'input_ids' in input_names:
            # NLP model: use input_ids and attention_mask
            seq_len = getattr(self._args, 'seq_length', 512)
            input_ids = np.random.randint(0, 30000, (self._args.batch_size, seq_len)).astype(np.int64)
            attention_mask = np.ones((self._args.batch_size, seq_len), dtype=np.int64)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            # Default for in-house torchvision models: use 'input' (batch_size, 3, 224, 224)
            input_tensor = np.random.randn(self._args.batch_size, 3, 224, 224).astype(dtype=precision)
            inputs = {'input': input_tensor}

        for i in range(self._args.num_warmup):
            ort_sess.run(None, inputs)

        elapse_times = list()
        for i in range(self._args.num_steps):
            start = time.time()
            ort_sess.run(None, inputs)
            end = time.time()
            elapse_times.append((end - start) * 1000)

        return elapse_times


BenchmarkRegistry.register_benchmark(
    'ort-inference',
    ORTInferenceBenchmark,
    platform=Platform.CUDA,
)
