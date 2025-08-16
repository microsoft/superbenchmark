# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Llama model benchmarks."""

import os
import pytest
import torch
import numpy as np
import logging
from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode, Precision
from superbench.benchmarks.model_benchmarks.pytorch_llama import PytorchLlama
import tempfile
import json
# To run this test with deterministic cuBLAS from the shell (set before CUDA init):
# CUBLAS_WORKSPACE_CONFIG=:4096:8 SB_LOG_LEVEL=INFO \
#   pytest -q tests/benchmarks/model_benchmarks/test_pytorch_llama.py -v

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_7b():
    """Test pytorch-llama2-7b benchmark for fp16 train and inference."""
    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 \
            --model_action train inference',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchLlama))
    assert (benchmark.name == 'pytorch-llama2-7b')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of llama2 7b model.
    assert (benchmark._args.hidden_size == 4096)
    assert (benchmark._args.num_hidden_layers == 32)
    assert (benchmark._args.num_attention_heads == 32)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 100)
    assert (benchmark._args.seq_len == 32)
    assert (benchmark._args.num_warmup == 1)
    assert (benchmark._args.num_steps == 2)

    # Test Dataset.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    for metric in [
        'fp16_train_step_time', 'fp16_train_throughput', 'fp16_inference_step_time', 'fp16_inference_throughput'
    ]:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_periodic_and_logging_combined(caplog, monkeypatch):
    """Single run to verify periodic fingerprint logs, in-memory recording, and log-file generation.

    - Enables strict determinism envs to enforce deterministic algorithms (and periodic fingerprints still log).
    - Runs with --deterministic --random_seed 42 and num_steps=100 to hit the cadence at step 100.
    - Enables --generate-log with a temp path and validates the file contents.
    - Confirms in-memory recording of losses and periodic fingerprints.
    - Confirms INFO logs contain the expected Loss/ActMean lines at step 100.
    """

    print("IN TEST")
    # Enable strict determinism if possible (must be before first CUDA init)
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        print("IN IF")
        monkeypatch.setenv('SB_STRICT_DETERMINISM', '1')
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # If CUDA is already initialized by a previous test, we cannot enable strict
    # determinism here as CUBLAS_WORKSPACE_CONFIG will be ignored. The test does
    # not require strict mode; it only validates logging and bookkeeping.

    caplog.set_level(logging.INFO, logger='superbench')

    log_path = tempfile.mktemp(suffix='.json')
    parameters = (
        '--hidden_size 128 --num_hidden_layers 2 --num_attention_heads 4 --num_key_value_heads 4 '
        '--intermediate_size 512 --batch_size 1 --seq_len 16 --num_warmup 1 --num_steps 100 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train '
        f'--generate-log --log-path {log_path}'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    try:
        assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

        # Check that the parameters related to determinism are set
        assert(benchmark._args.deterministic == True)
        assert(benchmark._args.random_seed == 42)
        assert(benchmark._args.generate_log == True)

        # Expect one loss and one activation fingerprint log at step 100 (cadence = 100)
        messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
        assert any('Loss at step 100:' in m for m in messages)
        assert any('ActMean at step 100:' in m for m in messages)

        # Check that losses are recorded in-memory
        assert hasattr(benchmark, '_model_run_losses')
        assert isinstance(benchmark._model_run_losses, list)
        assert len(benchmark._model_run_losses) > 0

        # Check that periodic fingerprints are recorded in-memory
        assert hasattr(benchmark, '_model_run_periodic')
        periodic = benchmark._model_run_periodic
        assert isinstance(periodic, dict)
        assert 'loss' in periodic and 'act_mean' in periodic and 'step' in periodic
        assert len(periodic['loss']) > 0
        assert len(periodic['act_mean']) > 0
        assert len(periodic['step']) > 0

        # Log-file generation and contents
        assert os.path.exists(log_path)
        assert benchmark._args.generate_log is True
        with open(log_path, 'r') as f:
            data = json.load(f)
        assert 'schema_version' in data
        assert 'metadata' in data
        assert 'per_step_fp32_loss' in data
        assert 'fingerprints' in data
        assert isinstance(data['per_step_fp32_loss'], list)
        assert isinstance(data['fingerprints'], dict)
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_nondeterministic_defaults():
    """Run in normal (non-deterministic) mode and assert new params are unset.

    Verifies that without passing determinism or logging flags:
    - args.deterministic is False
    - args.generate_log is False
    - args.log_path is None
    - args.compare_log is None
    - periodic fingerprints are present but empty (no entries when not deterministic)
    """
    parameters = (
        '--hidden_size 128 --num_hidden_layers 2 --num_attention_heads 4 --num_key_value_heads 4 '
        '--intermediate_size 512 --batch_size 1 --seq_len 16 --num_warmup 1 --num_steps 5 '
        '--precision float32 --sample_count 2 --model_action train'
    )
    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS
    args = benchmark._args
    assert args.deterministic is False
    assert getattr(args, 'generate_log', False) is False
    assert getattr(args, 'log_path', None) is None
    assert getattr(args, 'compare_log', None) is None

    # Periodic fingerprints should exist but be empty when not running in deterministic mode
    assert hasattr(benchmark, '_model_run_periodic')
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict)
    for key in ('loss', 'act_mean', 'step'):
        assert key in periodic
        assert len(periodic[key]) == 0
