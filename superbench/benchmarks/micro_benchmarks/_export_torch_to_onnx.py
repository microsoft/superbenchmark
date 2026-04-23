# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Export PyTorch models to ONNX format."""

from pathlib import Path

from packaging import version
import torch.hub
import torch.onnx
import torchvision.models

import traceback

from superbench.common.utils import logger


class torch2onnxExporter():
    """PyTorch model to ONNX exporter."""
    def __init__(self):
        """Constructor."""
        from transformers import BertConfig, GPT2Config, LlamaConfig
        from superbench.benchmarks.model_benchmarks.pytorch_bert import BertBenchmarkModel
        from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import GPT2BenchmarkModel
        from superbench.benchmarks.model_benchmarks.pytorch_lstm import LSTMBenchmarkModel
        from superbench.benchmarks.model_benchmarks.pytorch_llama import LlamaBenchmarkModel
        from superbench.benchmarks.model_benchmarks.pytorch_mixtral import MixtralBenchmarkModel

        self.num_classes = 100
        self.lstm_input_size = 256
        self.benchmark_models = {
            'lstm':
            lambda: LSTMBenchmarkModel(
                self.lstm_input_size,
                1024,
                8,
                False,
                self.num_classes,
            ),
            'bert-base':
            lambda: BertBenchmarkModel(
                BertConfig(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                ),
                self.num_classes,
            ),
            'bert-large':
            lambda: BertBenchmarkModel(
                BertConfig(
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                ),
                self.num_classes,
            ),
            'gpt2-small':
            lambda: GPT2BenchmarkModel(
                GPT2Config(
                    n_embd=768,
                    n_layer=12,
                    n_head=12,
                ),
                self.num_classes,
            ),
            'gpt2-medium':
            lambda: GPT2BenchmarkModel(
                GPT2Config(
                    n_embd=1024,
                    n_layer=24,
                    n_head=16,
                ),
                self.num_classes,
            ),
            'gpt2-large':
            lambda: GPT2BenchmarkModel(
                GPT2Config(
                    n_embd=1280,
                    n_layer=36,
                    n_head=20,
                ),
                self.num_classes,
            ),
            'gpt2-xl':
            lambda: GPT2BenchmarkModel(
                GPT2Config(
                    n_embd=1600,
                    n_layer=48,
                    n_head=25,
                ),
                self.num_classes,
            ),
            'llama2-7b':
            lambda: LlamaBenchmarkModel(
                LlamaConfig(
                    hidden_size=4096,
                    num_hidden_layers=32,
                    num_attention_heads=32,
                    num_key_value_heads=32,
                    intermediate_size=11008,
                ),
                self.num_classes,
            ),
            'llama2-13b':
            lambda: LlamaBenchmarkModel(
                LlamaConfig(
                    hidden_size=5120,
                    num_hidden_layers=40,
                    num_attention_heads=40,
                    num_key_value_heads=40,
                    intermediate_size=13824,
                ),
                self.num_classes,
            ),
            'llama2-70b':
            lambda: LlamaBenchmarkModel(
                LlamaConfig(
                    hidden_size=8192,
                    num_hidden_layers=80,
                    num_attention_heads=64,
                    num_key_value_heads=8,
                    intermediate_size=28672,
                ),
                self.num_classes,
            ),
        }

        # Only include Mixtral models if MixtralBenchmarkModel is available
        if MixtralBenchmarkModel is not None:
            from transformers import MixtralConfig
            self.benchmark_models.update(
                {
                    'mixtral-8x7b':
                    lambda: MixtralBenchmarkModel(
                        MixtralConfig(
                            hidden_size=4096,
                            num_hidden_layers=32,
                            num_attention_heads=32,
                            num_key_value_heads=8,
                            intermediate_size=14336,
                            max_position_embeddings=32768,
                            router_aux_loss_coef=0.02,
                        ),
                        self.num_classes,
                    ),
                    'mixtral-8x22b':
                    lambda: MixtralBenchmarkModel(
                        MixtralConfig(
                            hidden_size=6144,
                            num_hidden_layers=56,
                            num_attention_heads=48,
                            num_key_value_heads=8,
                            intermediate_size=16384,
                            max_position_embeddings=65536,
                            router_aux_loss_coef=0.001,
                        ),
                        self.num_classes,
                    ),
                }
            )

        self._onnx_model_path = Path(torch.hub.get_dir()) / 'onnx'
        self._onnx_model_path.mkdir(parents=True, exist_ok=True)

    def check_torchvision_model(self, model_name):
        """Check whether can export the torchvision model with given name.

        Args:
            model_name (str): Name of torchvision model to check.

        Returns:
            bool: True if the model can be exported, False otherwise.
        """
        if hasattr(torchvision.models, model_name):
            return True
        return False

    def check_benchmark_model(self, model_name):
        """Check whether can export the benchmark model with given name.

        Args:
            model_name (str): Name of benchmark model to check.

        Returns:
            bool: True if the model can be exported, False otherwise.
        """
        if model_name in self.benchmark_models:
            return True
        return False

    def export_torchvision_model(self, model_name, batch_size=1):
        """Export the torchvision model with given name.

        Args:
            model_name (str): Name of torchvision model to export.
            batch_size (int): Batch size of input. Defaults to 1.

        Returns:
            str: Exported ONNX model file name.
        """
        if not self.check_torchvision_model(model_name):
            return ''
        file_name = str(self._onnx_model_path / (model_name + '.onnx'))
        # the parameter 'pretrained' is deprecated since 0.13 in torchvision
        args = {'pretrained': False} if version.parse(torchvision.__version__) < version.parse('0.13') else {}
        model = getattr(torchvision.models, model_name)(**args).eval().cuda()
        dummy_input = torch.randn((batch_size, 3, 224, 224), device='cuda')
        torch.onnx.export(
            model,
            dummy_input,
            file_name,
            opset_version=14,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {
                    0: 'batch_size',
                },
                'output': {
                    0: 'batch_size',
                }
            },
        )

        del model
        del dummy_input
        torch.cuda.empty_cache()
        return file_name

    def export_benchmark_model(self, model_name, batch_size=1, seq_length=512):
        """Export the benchmark model with given name.

        Args:
            model_name (str): Name of benchmark model to export.
            batch_size (int): Batch size of input. Defaults to 1.
            seq_length (int): Sequence length of input. Defaults to 512.

        Returns:
            str: Exported ONNX model file name.
        """
        if not self.check_benchmark_model(model_name):
            return
        file_name = str(self._onnx_model_path / (model_name + '.onnx'))
        model = self.benchmark_models[model_name]().eval().cuda()
        dummy_input = torch.ones((batch_size, seq_length), dtype=torch.int64, device='cuda')
        if model_name == 'lstm':
            dummy_input = torch.ones((batch_size, seq_length, self.lstm_input_size), device='cuda')
        torch.onnx.export(
            model,
            dummy_input,
            file_name,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {
                    0: 'batch_size',
                    1: 'seq_length',
                },
                'output': {
                    0: 'batch_size',
                }
            },
        )

        del model
        del dummy_input
        torch.cuda.empty_cache()
        return file_name

    def export_huggingface_model(self, model, model_name, batch_size=1, seq_length=512, output_dir=None):
        """Export a HuggingFace model to ONNX format.

        Args:
            model: HuggingFace model instance to export.
            model_name (str): Name for the exported ONNX model file.
            batch_size (int): Batch size of input. Defaults to 1.
            seq_length (int): Sequence length of input. Defaults to 512.
            output_dir (str): Output directory path. If None, uses default path.

        Returns:
            str: Exported ONNX model file path, or empty string if export fails.
        """
        try:
            # Use custom output directory if provided
            output_path = Path(output_dir) if output_dir else self._onnx_model_path
            file_name = str(output_path / (model_name + '.onnx'))

            # Put model in eval mode and move to CUDA if available
            model.eval()

            # Disable cache to avoid DynamicCache issues with ONNX export
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False

            if torch.cuda.is_available():
                model = model.cuda()

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Get model's dtype for inputs
            model_dtype = next(model.parameters()).dtype

            # Detect model type and create appropriate inputs
            # Vision models use pixel_values, NLP models use input_ids
            # Use HuggingFace's main_input_name property for automatic detection
            main_input = getattr(model, 'main_input_name', 'input_ids')
            is_vision_model = main_input == 'pixel_values'

            if is_vision_model:
                # Vision models: use pixel_values (batch_size, channels, height, width)
                # Standard ImageNet size is 224x224, 3 channels
                # Match the dtype of the model
                dummy_input = torch.randn(batch_size, 3, 224, 224, dtype=model_dtype, device=device)
                input_names = ['pixel_values']
                dynamic_axes = {'pixel_values': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

                # Wrapper for vision models
                class VisionModelWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, pixel_values):
                        outputs = self.model(pixel_values=pixel_values)
                        if hasattr(outputs, 'logits'):
                            return outputs.logits
                        elif hasattr(outputs, 'last_hidden_state'):
                            return outputs.last_hidden_state
                        else:
                            return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                wrapped_model = VisionModelWrapper(model)
                export_args = (dummy_input, )
            else:
                # NLP models: use input_ids and attention_mask
                dummy_input = torch.ones((batch_size, seq_length), dtype=torch.int64, device=device)
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.int64, device=device)
                input_names = ['input_ids', 'attention_mask']
                dynamic_axes = {
                    'input_ids': {
                        0: 'batch_size',
                        1: 'seq_length'
                    },
                    'attention_mask': {
                        0: 'batch_size',
                        1: 'seq_length'
                    },
                    'output': {
                        0: 'batch_size'
                    },
                }

                # Wrapper for NLP models
                class NLPModelWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, input_ids, attention_mask):
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        if hasattr(outputs, 'logits'):
                            return outputs.logits
                        elif hasattr(outputs, 'last_hidden_state'):
                            return outputs.last_hidden_state
                        else:
                            return outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                wrapped_model = NLPModelWrapper(model)
                export_args = (dummy_input, attention_mask)

            # Export to ONNX for large models (>2GB), use external data format
            model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
            use_external_data = model_size_gb > 2.0

            if use_external_data:
                logger.info(f'Model size is {model_size_gb:.2f}GB, using external data format for ONNX export')

            torch.onnx.export(
                wrapped_model,
                export_args,
                file_name,
                opset_version=14,
                do_constant_folding=True,
                input_names=input_names,
                output_names=['output'],
                dynamic_axes=dynamic_axes,
            )

            # If using external data, convert to external data format
            if use_external_data:
                import onnx
                from onnx.external_data_helper import convert_model_to_external_data

                onnx_model = onnx.load(file_name)
                external_data_path = model_name + '_data.bin'
                convert_model_to_external_data(
                    onnx_model,
                    all_tensors_to_one_file=True,
                    location=external_data_path,
                    size_threshold=1024,
                    convert_attribute=False
                )
                onnx.save(onnx_model, file_name)
                logger.info(f'Converted ONNX model to external data format: {external_data_path}')

            # Clean up
            del dummy_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return file_name

        except Exception as e:
            logger.error(f'Failed to export HuggingFace model to ONNX: {str(e)}')
            logger.error(traceback.format_exc())
            return ''
