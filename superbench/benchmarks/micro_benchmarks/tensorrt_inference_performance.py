# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""TensorRT inference micro-benchmark."""

import os
import re
import subprocess
from pathlib import Path

import torch

from superbench.common.utils import logger
from superbench.benchmarks import BenchmarkRegistry, Platform, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke
from superbench.benchmarks.micro_benchmarks._export_torch_to_onnx import torch2onnxExporter
from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import (
    HuggingFaceModelLoader,
    validate_model_identifier,
)


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
            help='Source of the model: in-house (default) or huggingface.',
        )

        self._parser.add_argument(
            '--model_identifier',
            type=str,
            default=None,
            required=False,
            help='Model identifier for HuggingFace models (e.g., bert-base-uncased).',
        )

        self._parser.add_argument(
            '--allow_remote_code',
            action='store_true',
            default=False,
            required=False,
            help='Allow HuggingFace to execute model-repo Python (trust_remote_code=True). '
            'SECURITY: enables RCE from --model_identifier. Pin --revision <sha> when used.',
        )

    @staticmethod
    def __detect_workspace_flag(bin_path: str) -> str:
        """Return the trtexec workspace flag supported by the installed binary.

        Args:
            bin_path: Absolute path to the trtexec binary.

        Returns:
            ``'--memPoolSize=workspace:8192M'`` on TensorRT >= 8.4,
            ``'--workspace=8192'`` on older runtimes or when probing fails.
        """
        modern = '--memPoolSize=workspace:8192M'
        legacy = '--workspace=8192'
        try:
            proc = subprocess.run([bin_path, '--help'], capture_output=True, text=True, timeout=10, check=False)
            help_text = (proc.stdout or '') + (proc.stderr or '')
            if '--memPoolSize' in help_text:
                return modern
            logger.warning(
                'trtexec at %s does not advertise --memPoolSize; falling back to --workspace=8192 '
                '(TensorRT < 8.4 detected).', bin_path
            )
            return legacy
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(
                'Could not probe trtexec at %s for --memPoolSize support (%s); using --workspace=8192.',
                bin_path,
                e,
            )
            return legacy

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        self.__bin_path = str(Path(self._args.bin_dir) / self._bin_name)
        # Pick the right workspace flag for the installed trtexec. --memPoolSize was
        # introduced in TensorRT 8.4; older runtimes (TRT 8.0-8.3, still found in
        # some CUDA 11.x base images) only accept the deprecated-but-still-supported
        # --workspace=. Probe once here and reuse for every model.
        self.__workspace_flag = self.__detect_workspace_flag(self.__bin_path)

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
                self.__workspace_flag,
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
        from transformers import AutoConfig

        if not self._args.model_identifier:
            logger.error('--model_identifier is required when using --model_source huggingface')
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            return False

        # Reject malformed / path-like identifiers up front, before any network or disk activity.
        try:
            validate_model_identifier(self._args.model_identifier)
        except ValueError as e:
            logger.error(str(e))
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            return False

        allow_remote_code = bool(getattr(self._args, 'allow_remote_code', False))

        # Reject INT8 on the HuggingFace path: the current pipeline emits `--int8` to
        # trtexec without `--calib=<file>` and without a Q/DQ-embedded ONNX, so trtexec
        # would fall back to fake dynamic ranges and report misleading latencies.
        if str(getattr(self._args, 'precision', '')).lower() == 'int8':
            logger.error(
                'TensorRT --precision int8 on HuggingFace models is not supported: '
                'no calibration data / Q-DQ ONNX is generated, so reported latencies '
                'would not represent a correctly-calibrated INT8 engine. '
                'Use --precision fp16 or fp32, or run ORT INT8 quantization first.'
            )
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            return False

        try:
            # Step 1: Pre-download memory check — download only the config (a few KB)
            # and estimate whether the full model will fit in GPU memory.
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
            load_kwargs = {}
            if hf_token:
                load_kwargs['token'] = hf_token

            hf_config = AutoConfig.from_pretrained(
                self._args.model_identifier, trust_remote_code=allow_remote_code, **load_kwargs
            )
            precision_str = self._args.precision    # already a string: 'fp16', 'fp32', 'int8'
            fits, param_m, est_gb, avail_gb = HuggingFaceModelLoader.check_memory_fits(
                self._args.model_identifier, hf_config, precision_str, mode='inference', token=hf_token
            )
            if not fits:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
                return False

            # Step 2: Download, export to ONNX, and build the trtexec command.
            return self._build_trtexec_command_for_hf(hf_token, allow_remote_code)

        except Exception as e:
            logger.error(f'Failed to prepare HuggingFace model: {str(e)}')
            import traceback
            logger.error(traceback.format_exc())
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            return False

    def _build_trtexec_command_for_hf(self, hf_token, allow_remote_code):
        """Download HF model, export to ONNX, derive input shapes, and append the trtexec command.

        Args:
            hf_token (str | None): HuggingFace token, or None.
            allow_remote_code (bool): Whether to allow trust_remote_code on load.

        Returns:
            bool: True on success; False (with return code set) on failure.
        """
        # Get GPU rank to create unique file paths and avoid race conditions
        # when multiple processes export the same model simultaneously
        gpu_rank = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        proc_rank = os.getenv('PROC_RANK', gpu_rank)

        # Create model source config - load on CPU to avoid accelerate dispatching
        # model across multiple GPUs which causes device mismatch during ONNX export.
        # TensorRT handles precision internally via --fp16/--int8 flags,
        # so the ONNX model is always exported in float32.
        model_config = ModelSourceConfig(
            source='huggingface',
            identifier=self._args.model_identifier,
            hf_token=hf_token,
            torch_dtype='float32',
            device_map=None,
        )

        logger.info(f'Loading HuggingFace model: {self._args.model_identifier}')

        # Load model from HuggingFace on CPU
        loader = HuggingFaceModelLoader(allow_remote_code=allow_remote_code)
        hf_model, hf_config, _ = loader.load_model_from_config(model_config, device='cpu')
        self._hf_config = hf_config
        exporter = torch2onnxExporter()

        model_name = self._args.model_identifier.replace('/', '_')

        # Prepare output path - use proc_rank subdirectory to avoid race conditions
        # when multiple processes export the same model simultaneously
        output_dir = str(Path(torch.hub.get_dir()) / 'checkpoints' / f'trt_rank_{proc_rank}')
        os.makedirs(output_dir, exist_ok=True)

        # Defense-in-depth: confirm resolved output path stays inside the rank directory
        # even though validate_model_identifier already rejected '..' / '\\' / control chars.
        proc_root = Path(output_dir).resolve()
        resolved_out = (Path(output_dir) / f'{model_name}.onnx').resolve()
        if proc_root not in resolved_out.parents:
            logger.error(f'Refusing to write ONNX outside rank dir: {resolved_out} not under {proc_root}')
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            return False

        onnx_path = exporter.export_huggingface_model(
            model=hf_model,
            model_name=model_name,
            batch_size=self._args.batch_size,
            seq_length=self._args.seq_length,
            output_dir=output_dir,
        )

        if not onnx_path:
            logger.error(f'Failed to export {self._args.model_identifier} to ONNX')
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE)
            return False

        input_shapes = self._derive_trt_input_shapes(onnx_path)

        # Build TensorRT command with correct input name
        args = [
            self.__bin_path,
            f'--onnx={onnx_path}',
            f'--optShapes={input_shapes}',
            self.__workspace_flag,
            None if self._args.precision == 'fp32' else f'--{self._args.precision}',
            f'--iterations={self._args.iterations}',
            '--percentile=99',
        ]
        self._commands.append(' '.join(filter(None, args)))

        # Store model name for result processing
        self._args.pytorch_models = [self._args.model_identifier.replace('/', '_')]

        logger.info('Successfully prepared HuggingFace model for TensorRT inference')
        return True

    def _derive_trt_input_shapes(self, onnx_path):
        """Inspect the exported ONNX graph and produce the trtexec ``--optShapes`` value.

        Args:
            onnx_path (str): Path to the exported ONNX file.

        Returns:
            str: Comma-separated ``name:DxDxD...`` string suitable for trtexec ``--optShapes``.
        """
        # Pass load_external_data=False because we only need graph input metadata;
        # the default True would materialize all sidecar tensors and OOM on the
        # >2GB external-data models that this branch was written for.
        import onnx as onnx_lib
        onnx_model = onnx_lib.load(onnx_path, load_external_data=False)

        # Filter out initializers from graph.input to get only runtime inputs
        initializer_names = {init.name for init in onnx_model.graph.initializer}
        runtime_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initializer_names]
        if not runtime_inputs:
            raise ValueError(f'No runtime inputs found in exported ONNX model: {onnx_path}')

        # Get the first runtime input to determine shape and name
        input_name = runtime_inputs[0].name

        # Vision models typically have 4D input (batch, channels, height, width)
        # NLP models typically have 2D input (batch, sequence)
        if input_name == 'pixel_values' or len(runtime_inputs[0].type.tensor_type.shape.dim) == 4:
            return self._derive_vision_input_shape(runtime_inputs[0], input_name)
        return self._derive_nlp_input_shapes(runtime_inputs)

    def _derive_vision_input_shape(self, runtime_input, input_name):
        """Build the optShapes string for a vision model with 4D input."""
        dims = runtime_input.type.tensor_type.shape.dim
        # dims[0] is batch, dims[1:] are C, H, W
        c_dim = dims[1].dim_value if dims[1].dim_value > 0 else None
        h_dim = dims[2].dim_value if dims[2].dim_value > 0 else None
        w_dim = dims[3].dim_value if dims[3].dim_value > 0 else None

        # Fall back to HF config metadata when ONNX dims are dynamic/unknown
        if hasattr(self, '_hf_config'):
            channels = c_dim or getattr(self._hf_config, 'num_channels', 3)
            image_size = getattr(self._hf_config, 'image_size', 224)
            if isinstance(image_size, (list, tuple)):
                height = h_dim or image_size[0]
                width = w_dim or image_size[1]
            else:
                height = h_dim or image_size
                width = w_dim or image_size
        else:
            channels = c_dim or 3
            height = h_dim or 224
            width = w_dim or 224

        return f'{input_name}:{self._args.batch_size}x{channels}x{height}x{width}'

    def _derive_nlp_input_shapes(self, runtime_inputs):
        """Build the optShapes string for an NLP model (2D batch x sequence inputs)."""
        seq_len = getattr(self._args, 'seq_length', 512)
        shapes_list = []
        for inp in runtime_inputs:
            inp_name = inp.name
            num_dims = len(inp.type.tensor_type.shape.dim)
            if num_dims == 4:
                # 4D input (rare for NLP, but handle it)
                shapes_list.append(f'{inp_name}:{self._args.batch_size}x1x{seq_len}x{seq_len}')
            else:
                # Default to 2D batch x sequence
                shapes_list.append(f'{inp_name}:{self._args.batch_size}x{seq_len}')
        return ','.join(shapes_list)

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
