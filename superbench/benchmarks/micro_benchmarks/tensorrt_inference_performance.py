# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TensorRT inference micro-benchmark."""

import re
from pathlib import Path

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke
from superbench.benchmarks.micro_benchmarks._export_torch_to_onnx import torch2onnxExporter
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import HuggingFaceModelLoader


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

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = str(Path(self._args.bin_dir) / self._bin_name)

        # Handle HuggingFace models if specified
        if self._args.model_source == 'huggingface':
            return self._preprocess_huggingface_models()

        # Original in-house model processing
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
                f'--optShapes=input:{input_shape}',
                f'--memPoolSize=workspace:8192M',
                None if self._args.precision == 'fp32' else f'--{self._args.precision}',
                # inference options
                f'--iterations={self._args.iterations}',
                # reporting options
                '--percentile=99',
            ]   # yapf: disable
            self._commands.append(' '.join(filter(None, args)))

        return True

    def _preprocess_huggingface_models(self):
        """Preprocess HuggingFace models for TensorRT inference.

        Returns:
            bool: True if preprocessing succeeds.
        """
        import torch
        import os
        import time
        from transformers import AutoConfig

        if not self._args.model_identifier:
            logger.error('--model_identifier is required when using --model_source huggingface')
            return False

        try:
            # Step 1: Pre-download memory check — download only the config (a few KB)
            # and estimate whether the full model will fit in GPU memory.
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            load_kwargs = {}
            if hf_token:
                load_kwargs['token'] = hf_token

            hf_config = AutoConfig.from_pretrained(
                self._args.model_identifier, **load_kwargs
            )
            precision_str = self._args.precision    # already a string: 'fp16', 'fp32', 'int8'
            fits, param_m, est_gb, avail_gb = HuggingFaceModelLoader.check_memory_fits(
                self._args.model_identifier, hf_config, precision_str, mode='inference', token=hf_token
            )
            if not fits:
                return False

            # Step 2: Download and load the full model
            logger.info(f'Loading HuggingFace model: {self._args.model_identifier}')

            loader = HuggingFaceModelLoader()
            hf_model, hf_config, tokenizer = loader.load_model(
                model_identifier=self._args.model_identifier,
                torch_dtype='float32',
                device='cpu',
                device_map=None,
            )
            exporter = torch2onnxExporter()

            # Get process rank for unique directory
            proc_rank = os.environ.get('PROC_RANK', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
            output_dir = f'/tmp/tensorrt_onnx_rank_{proc_rank}'
            os.makedirs(output_dir, exist_ok=True)

            onnx_path = exporter.export_huggingface_model(
                model=hf_model,
                model_name=self._args.model_identifier.replace('/', '_'),
                batch_size=self._args.batch_size,
                seq_length=getattr(self._args, 'seq_length', 512),
                output_dir=output_dir,
            )

            if not onnx_path:
                logger.error(f'Failed to export {self._args.model_identifier} to ONNX')
                return False

            # Determine input shape based on model type by checking ONNX file
            import onnx as onnx_lib
            onnx_model = onnx_lib.load(onnx_path)

            # Get the first input to determine shape and name
            input_name = onnx_model.graph.input[0].name

            # Vision models typically have 4D input (batch, channels, height, width)
            # NLP models typically have 2D input (batch, sequence)
            if input_name == 'pixel_values' or len(onnx_model.graph.input[0].type.tensor_type.shape.dim) == 4:
                # Vision model: batch x channels x height x width
                input_shapes = f'{input_name}:{self._args.batch_size}x3x224x224'
            else:
                # NLP model: batch x sequence - need to specify all inputs with same batch and seq length
                seq_len = getattr(self._args, 'seq_length', 512)
                shapes_list = []
                for inp in onnx_model.graph.input:
                    inp_name = inp.name
                    num_dims = len(inp.type.tensor_type.shape.dim)
                    if num_dims == 2:
                        # Standard 2D input: batch x sequence
                        shapes_list.append(f'{inp_name}:{self._args.batch_size}x{seq_len}')
                    elif num_dims == 4:
                        # 4D input (rare for NLP, but handle it)
                        shapes_list.append(f'{inp_name}:{self._args.batch_size}x1x{seq_len}x{seq_len}')
                    else:
                        # Default to 2D
                        shapes_list.append(f'{inp_name}:{self._args.batch_size}x{seq_len}')
                input_shapes = ','.join(shapes_list)

            # Build TensorRT command with correct input name
            args = [
                self.__bin_path,
                f'--onnx={onnx_path}',
                f'--optShapes={input_shapes}',
                f'--memPoolSize=workspace:8192M',
                None if self._args.precision == 'fp32' else f'--{self._args.precision}',
                f'--iterations={self._args.iterations}',
                '--percentile=99',
            ]
            self._commands.append(' '.join(filter(None, args)))

            # Store model name for result processing
            self._args.pytorch_models = [self._args.model_identifier.replace('/', '_')]

            logger.info(f'Successfully prepared HuggingFace model for TensorRT inference')
            return True

        except Exception as e:
            logger.error(f'Failed to prepare HuggingFace model: {str(e)}')
            import traceback
            logger.error(traceback.format_exc())
            return False

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
                    lats = re.findall(r'(\d+\.*\d*) ms', line)
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
                        float(re.findall(r'mean = (\d+\.*\d*) ms', line)[0]),
                    )
                    self._result.add_result(
                        f'{model}_{tm}_time_99',
                        float(re.findall(r'\(99\%\) = (\d+\.*\d*) ms', line)[0]),
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
