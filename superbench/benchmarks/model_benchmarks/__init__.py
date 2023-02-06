# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A module containing all the e2e model related benchmarks."""

from superbench.common.utils import LazyImport

ModelBenchmark = LazyImport('superbench.benchmarks.model_benchmarks.model_base.ModelBenchmark')
PytorchBERT = LazyImport('superbench.benchmarks.model_benchmarks.model_benchmarks.pytorch_bert.PytorchBERT')
PytorchGPT2 = LazyImport('superbench.benchmarks.model_benchmarks.model_benchmarks.pytorch_gpt2.PytorchGPT2')
PytorchCNN = LazyImport('superbench.benchmarks.model_benchmarks.pytorch_cnn.PytorchCNN')
PytorchLSTM = LazyImport('superbench.benchmarks.model_benchmarks.pytorch_lstm.PytorchLSTM')


__all__ = ['ModelBenchmark', 'PytorchBERT', 'PytorchGPT2', 'PytorchCNN', 'PytorchLSTM']
