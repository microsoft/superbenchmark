# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for BERT model benchmarks."""

import os
import logging
import pytest
import torch
from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_bert import PytorchBERT
import json
import tempfile

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_bert_base():
    """Test pytorch-bert-base benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'bert-base',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
            --model_action train inference',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchBERT))
    assert (benchmark.name == 'pytorch-bert-base')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of resnet101 model.
    assert (benchmark._args.hidden_size == 768)
    assert (benchmark._args.num_hidden_layers == 12)
    assert (benchmark._args.num_attention_heads == 12)
    assert (benchmark._args.intermediate_size == 3072)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 5)
    assert (benchmark._args.seq_len == 8)
    assert (benchmark._args.num_warmup == 2)
    assert (benchmark._args.num_steps == 4)

    # Check dataset scale.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for metric in [
        'fp32_train_step_time', 'fp32_train_throughput', 'fp16_train_step_time', 'fp16_train_throughput',
        'fp32_inference_step_time', 'fp32_inference_throughput', 'fp16_inference_step_time', 'fp16_inference_throughput'
    ]:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)



@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_bert_periodic_and_logging_combined(caplog, monkeypatch):
    """Verify periodic fingerprint logs, in-memory recording, and log-file generation in a single run.

    - Enables strict determinism envs if CUDA not initialized (optional).
    - Runs with --deterministic --random_seed 42 and num_steps=100 to hit cadence at step 100.
    - Enables --generate-log with a temp path; validates file contents and in-memory bookkeeping.
    - Confirms INFO logs contain Loss/ActMean at step 100.
    """

    # Enable strict determinism if possible (must be before first CUDA init)
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        monkeypatch.setenv('SB_STRICT_DETERMINISM', '1')
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    caplog.set_level(logging.INFO, logger='superbench')

    log_path = tempfile.mktemp(suffix='.json')
    parameters = (
        '--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 4 '
        '--intermediate_size 1024 --batch_size 1 --seq_len 16 --num_warmup 1 --num_steps 100 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train '
        f'--generate-log --log-path {log_path}'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'bert-base', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    try:
        assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

        # Check determinism/logging args
        assert benchmark._args.deterministic is True
        assert benchmark._args.random_seed == 42
        assert benchmark._args.generate_log is True

        # Expect Loss/ActMean logs at step 100
        messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
        assert any('Loss at step 100:' in m for m in messages)
        assert any('ActMean at step 100:' in m for m in messages)

        # In-memory recording
        assert hasattr(benchmark, '_model_run_losses') and isinstance(benchmark._model_run_losses, list)
        assert len(benchmark._model_run_losses) > 0
        assert hasattr(benchmark, '_model_run_periodic') and isinstance(benchmark._model_run_periodic, dict)
        periodic = benchmark._model_run_periodic
        for key in ('loss', 'act_mean', 'step'):
            assert key in periodic
        assert len(periodic['loss']) > 0
        assert len(periodic['act_mean']) > 0
        assert len(periodic['step']) > 0

        # Log-file generation and contents
        assert os.path.exists(log_path)
        with open(log_path, 'r') as f:
            data = json.load(f)
        assert 'schema_version' in data
        assert 'metadata' in data
        assert 'per_step_fp32_loss' in data and isinstance(data['per_step_fp32_loss'], list)
        assert 'fingerprints' in data and isinstance(data['fingerprints'], dict)
        # Optional: verify step 100 present if any steps recorded
        fp = data['fingerprints']
        if 'step' in fp and isinstance(fp['step'], list) and len(fp['step']) > 0:
            assert 100 in fp['step']
            assert len(fp.get('loss', [])) == len(fp['step'])
            assert len(fp.get('act_mean', [])) == len(fp['step'])
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_bert_nondeterministic_defaults():
    """Run without determinism/logging flags and assert defaults are unset and periodic is empty."""
    parameters = (
        '--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 4 '
        '--intermediate_size 1024 --batch_size 1 --seq_len 16 --num_warmup 1 --num_steps 5 '
        '--precision float32 --sample_count 2 --model_action train'
    )
    context = BenchmarkRegistry.create_benchmark_context(
        'bert-base', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS
    args = benchmark._args
    assert args.deterministic is False
    assert getattr(args, 'generate_log', False) is False
    assert getattr(args, 'log_path', None) is None
    assert getattr(args, 'compare_log', None) is None

    # Periodic fingerprints exist but are empty when not deterministic
    assert hasattr(benchmark, '_model_run_periodic')
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict)
    for key in ('loss', 'act_mean', 'step'):
        assert key in periodic
        assert len(periodic[key]) == 0
