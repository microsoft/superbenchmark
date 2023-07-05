# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Export PyTorch models to ONNX format."""

from pathlib import Path

from packaging import version
import torch.hub
import torch.onnx
import torchvision.models
from transformers import BertConfig, GPT2Config

from superbench.benchmarks.model_benchmarks.pytorch_bert import BertBenchmarkModel
from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import GPT2BenchmarkModel
from superbench.benchmarks.model_benchmarks.pytorch_lstm import LSTMBenchmarkModel


class torch2onnxExporter():
    """PyTorch model to ONNX exporter."""
    def __init__(self):
        """Constructor."""
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
        }
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
            opset_version=10,
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
            opset_version=10,
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
