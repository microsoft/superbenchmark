# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BERT model benchmarks."""

from importlib import reload

from superbench.benchmarks import BenchmarkRegistry, Precision, Platform, Framework, BenchmarkContext
import superbench.benchmarks.model_benchmarks.pytorch_bert as pybert


def test_pytorch_bert_base():
    """Test pytorch-bert-base benchmark."""
    context = BenchmarkContext(
        'bert-base',
        Platform.CUDA,
        parameters='--batch_size=32 --num_classes=5 --seq_len=512',
        framework=Framework.PYTORCH
    )

    # Clean the registry and register the BERT benchmarks.
    BenchmarkRegistry.clean_benchmarks()
    reload(pybert)

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))
    assert (BenchmarkRegistry.check_parameters(context))

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
    samples_count = (benchmark._args.batch_size * (benchmark._args.num_warmup + benchmark._args.num_steps))
    assert (len(benchmark._dataset) == samples_count)

    # Test _create_model().
    assert (benchmark._create_model(Precision.FLOAT32) is True)
    assert (isinstance(benchmark._model, pybert.BertBenchmarkModel))

    BenchmarkRegistry.clean_benchmarks()


def test_pytorch_bert_large():
    """Test pytorch-bert-large benchmark."""
    context = BenchmarkContext(
        'bert-large',
        Platform.CUDA,
        parameters='--batch_size=32 --num_classes=5 --seq_len=512',
        framework=Framework.PYTORCH
    )

    # Clean the registry and register the BERT benchmarks.
    BenchmarkRegistry.clean_benchmarks()
    reload(pybert)

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))
    assert (BenchmarkRegistry.check_parameters(context))

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
    samples_count = (benchmark._args.batch_size * (benchmark._args.num_warmup + benchmark._args.num_steps))
    assert (len(benchmark._dataset) == samples_count)

    # Test _create_model().
    assert (benchmark._create_model(Precision.FLOAT32) is True)
    assert (isinstance(benchmark._model, pybert.BertBenchmarkModel))

    BenchmarkRegistry.clean_benchmarks()
