# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Export PyTorch models to ONNX format."""

from pathlib import Path

import torch.hub
import torch.onnx
import torchvision.models
from transformers import BertConfig, GPT2Config

from superbench.benchmarks.model_benchmarks.pytorch_bert import BertBenchmarkModel
from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import GPT2BenchmarkModel


class torch2onnxExporter():
    """PyTorch model to ONNX exporter."""
    def __init__(self):
        """Constructor."""
        transformers_num_classes = 100
        self.transformers = {
            'bert-base':
            BertBenchmarkModel(
                BertConfig(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                ),
                transformers_num_classes,
            ),
            'bert-large':
            BertBenchmarkModel(
                BertConfig(
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                ),
                transformers_num_classes,
            ),
            'gpt2-small':
            GPT2BenchmarkModel(
                GPT2Config(
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                ),
                transformers_num_classes,
            ),
            'gpt2-medium':
            GPT2BenchmarkModel(
                GPT2Config(
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                ),
                transformers_num_classes,
            ),
            'gpt2-large':
            GPT2BenchmarkModel(
                GPT2Config(
                    hidden_size=1280,
                    num_hidden_layers=36,
                    num_attention_heads=20,
                ),
                transformers_num_classes,
            ),
            'gpt2-xl':
            GPT2BenchmarkModel(
                GPT2Config(
                    hidden_size=1600,
                    num_hidden_layers=48,
                    num_attention_heads=25,
                ),
                transformers_num_classes,
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

    def check_transformers_model(self, model_name):
        """Check whether can export the transformers model with given name.

        Args:
            model_name (str): Name of transformers model to check.

        Returns:
            bool: True if the model can be exported, False otherwise.
        """
        if model_name in self.transformers:
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
        torch.onnx.export(
            getattr(torchvision.models, model_name).eval().cuda(),
            torch.randn(batch_size, 3, 224, 224, device='cuda'),
            file_name,
            opset_version=10,
            do_constant_folding=True,
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
        return file_name

    def export_transformers_model(self, model_name, batch_size=1, seq_length=512):
        """Export the transformers model with given name.

        Args:
            model_name (str): Name of transformers model to export.
            batch_size (int): Batch size of input. Defaults to 1.
            seq_length (int): Sequence length of input. Defaults to 512.

        Returns:
            str: Exported ONNX model file name.
        """
        if not self.check_transformers_model(model_name):
            return
        file_name = str(self._onnx_model_path / (model_name + '.onnx'))
        torch.onnx.export(
            self.transformers[model_name].eval().cuda(),
            torch.empty(batch_size, seq_length, dtype=torch.int64, device='cuda'),
            file_name,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['outputs'],
            dynamic_axes={
                'input': {
                    0: 'batch_size',
                    1: 'seq_length',
                },
                'outputs': {
                    0: 'batch_size',
                }
            },
        )
        return file_name
