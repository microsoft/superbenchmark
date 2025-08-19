# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for CNN model benchmarks."""

import os
import logging
import numpy as np
import pytest
import torch
import json
import tempfile

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_cnn import PytorchCNN


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_cnn_with_gpu():
    """Test pytorch cnn benchmarks with GPU."""
    run_pytorch_cnn(
        models=['resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19'],
        parameters='--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 \
            --model_action train inference',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp16_train_step_time', 'fp16_train_throughput',
            'fp32_inference_step_time', 'fp32_inference_throughput', 'fp16_inference_step_time',
            'fp16_inference_throughput'
        ]
    )


@decorator.pytorch_test
def test_pytorch_cnn_no_gpu():
    """Test pytorch cnn benchmarks with CPU."""
    run_pytorch_cnn(
        models=['resnet50'],
        parameters='--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 \
                --model_action train inference --precision float32 --no_gpu',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp32_inference_step_time', 'fp32_inference_throughput'
        ]
    )


def run_pytorch_cnn(models=[], parameters='', check_metrics=[]):
    """Run pytorch cnn benchmarks."""
    for model in models:
        context = BenchmarkRegistry.create_benchmark_context(
            model, platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
        )

        assert (BenchmarkRegistry.is_benchmark_context_valid(context))

        benchmark = BenchmarkRegistry.launch_benchmark(context)

        # Check basic information.
        assert (benchmark)
        assert (isinstance(benchmark, PytorchCNN))
        assert (benchmark.name == 'pytorch-' + model)
        assert (benchmark.type == BenchmarkType.MODEL)

        # Check predefined parameters of resnet101 model.
        assert (benchmark._args.model_type == model)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.batch_size == 1)
        assert (benchmark._args.image_size == 224)
        assert (benchmark._args.num_classes == 5)
        assert (benchmark._args.num_warmup == 2)
        assert (benchmark._args.num_steps == 4)

        # Check Dataset.
        assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

        # Check results and metrics.
        assert (benchmark.run_count == 1)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        for metric in check_metrics:
            assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
            assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
            assert (len(benchmark.result[metric]) == benchmark.run_count)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_cnn_periodic_and_logging_combined(caplog, monkeypatch):
    """Single run to verify periodic fingerprint logs, in-memory recording, and log-file generation."""

    # Enable strict determinism if possible (must be before first CUDA init)
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        monkeypatch.setenv('SB_STRICT_DETERMINISM', '1')
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    caplog.set_level(logging.INFO, logger='superbench')

    log_path = tempfile.mktemp(suffix='.json')
    parameters = (
        '--batch_size 1 --image_size 64 --num_classes 5 --num_warmup 1 --num_steps 100 '
        '--precision float32 --deterministic --random_seed 42 --model_action train '
        f'--generate-log --log-path {log_path} --check_frequency 10'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'resnet18', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    try:
        assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

        # Check that the parameters related to determinism are set
        assert benchmark._args.deterministic is True
        assert benchmark._args.random_seed == 42
        assert benchmark._args.generate_log is True
        assert benchmark._args.check_frequency == 10

        # Expect one loss and one activation fingerprint log at step 100 (cadence = 100)
        messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
        assert any('Loss at step 100:' in m for m in messages)
        assert any('ActMean at step 100:' in m for m in messages)

        # In-memory records
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
def test_pytorch_cnn_nondeterministic_defaults():
    """Run in normal (non-deterministic) mode and assert new params are unset and periodic empty."""
    parameters = (
        '--batch_size 1 --image_size 64 --num_classes 5 --num_warmup 1 --num_steps 5 '
        '--precision float32 --model_action train'
    )
    context = BenchmarkRegistry.create_benchmark_context(
        'resnet18', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS
    args = benchmark._args
    assert args.deterministic is False
    assert getattr(args, 'generate_log', False) is False
    assert getattr(args, 'log_path', None) is None
    assert getattr(args, 'compare_log', None) is None
    assert getattr(args, 'check_frequency', None) is 100

    # Periodic fingerprints should exist but be empty when not running in deterministic mode
    assert hasattr(benchmark, '_model_run_periodic')
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict)
    for key in ('loss', 'act_mean', 'step'):
        assert key in periodic
        assert len(periodic[key]) == 0
