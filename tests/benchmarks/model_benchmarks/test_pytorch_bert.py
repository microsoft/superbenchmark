# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BERT model benchmarks."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Precision, Platform, Framework
import superbench.benchmarks.model_benchmarks.pytorch_bert as pybert


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_bert_base():
    """Test pytorch-bert-base benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'bert-base',
        platform=Platform.CUDA,
        parameters='--batch_size=32 --num_classes=5 --seq_len=512',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark_name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (benchmark_name == 'pytorch-bert-base')

    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, context.platform)
    assert (benchmark_class == pybert.PytorchBERT)

    parameters = context.parameters
    if predefine_params:
        parameters = predefine_params + ' ' + parameters

    benchmark = benchmark_class(benchmark_name, parameters)
    assert (benchmark._preprocess() is True)

    # Predefined parameters of bert-base model.
    assert (benchmark._args.hidden_size == 768)
    assert (benchmark._args.num_hidden_layers == 12)
    assert (benchmark._args.num_attention_heads == 12)
    assert (benchmark._args.intermediate_size == 3072)

    # Parameters from BenchmarkContext.
    assert (benchmark._args.batch_size == 32)
    assert (benchmark._args.num_classes == 5)
    assert (benchmark._args.seq_len == 512)

    # Test Dataset.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Test _create_model().
    assert (benchmark._create_model(Precision.FLOAT32) is True)
    assert (isinstance(benchmark._model, pybert.BertBenchmarkModel))


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_bert_large():
    """Test pytorch-bert-large benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'bert-large',
        platform=Platform.CUDA,
        parameters='--batch_size=32 --num_classes=5 --seq_len=512',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark_name = BenchmarkRegistry._BenchmarkRegistry__get_benchmark_name(context)
    assert (benchmark_name == 'pytorch-bert-large')

    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, context.platform)
    assert (benchmark_class is pybert.PytorchBERT)

    parameters = context.parameters
    if predefine_params:
        parameters = predefine_params + ' ' + parameters

    benchmark = benchmark_class(benchmark_name, parameters)
    assert (benchmark._preprocess() is True)

    # Predefined parameters of bert-large model.
    assert (benchmark._args.hidden_size == 1024)
    assert (benchmark._args.num_hidden_layers == 24)
    assert (benchmark._args.num_attention_heads == 16)
    assert (benchmark._args.intermediate_size == 4096)

    # Parameters from BenchmarkContext.
    assert (benchmark._args.batch_size == 32)
    assert (benchmark._args.num_classes == 5)
    assert (benchmark._args.seq_len == 512)

    # Test Dataset.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Test _create_model().
    assert (benchmark._create_model(Precision.FLOAT32) is True)
    assert (isinstance(benchmark._model, pybert.BertBenchmarkModel))
