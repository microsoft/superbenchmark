# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for Megatron command forwarding and failure reporting."""

import os
from subprocess import CompletedProcess
from unittest import mock

import pytest

from superbench.benchmarks import Precision, ReturnCode
from superbench.benchmarks.result import BenchmarkResult
from superbench.benchmarks.model_benchmarks.megatron_gpt3 import MegatronGPT
from superbench.benchmarks.model_benchmarks.model_base import ModelBenchmark


def make_benchmark(parameters=''):
    """Build a command-only benchmark without initializing MPI or downloading data."""
    benchmark = MegatronGPT('megatron-gpt', parameters)
    benchmark.add_parser_arguments()
    valid, benchmark._args, _ = benchmark.parse_args()
    assert valid
    benchmark._distributed_args = '--nproc_per_node 2'
    benchmark._data_options = '--mock-data'
    return benchmark


def test_extra_arguments_forwarded():
    """Version-specific options must reach the selected Megatron checkout."""
    benchmark = make_benchmark('--extra "--vocab-size 128 --attention-backend unfused"')
    command = benchmark._megatron_command(Precision.BFLOAT16)
    assert command.endswith('--mock-data --vocab-size 128 --attention-backend unfused')


@pytest.mark.parametrize(
    'explicit,deepspeed,override,expected',
    [
        ('', False, '/rocm10/Megatron-LM', '/rocm10/Megatron-LM'),
        ('', False, '', '/legacy/third_party/Megatron/Megatron-LM'),
        ('', True, '/rocm10/Megatron-LM', '/legacy/third_party/Megatron/Megatron-DeepSpeed/'),
        ('/explicit', False, '/rocm10/Megatron-LM', '/explicit'),
    ]
)
def test_code_base_selection(explicit, deepspeed, override, expected):
    """The image override must preserve explicit paths and legacy/DeepSpeed defaults."""
    benchmark = make_benchmark('--batch_size 32 --num_gpus 2')
    benchmark._args.code_base = explicit
    benchmark._args.deepspeed = deepspeed
    benchmark._num_nodes = 1
    with mock.patch.dict(os.environ, {'SB_MICRO_PATH': '/legacy', 'SB_MEGATRON_PATH': override}):
        with mock.patch.object(ModelBenchmark, '_preprocess', return_value=True):
            with mock.patch('superbench.benchmarks.model_benchmarks.megatron_gpt3.os.path.exists', return_value=True):
                assert benchmark._preprocess()
    assert benchmark._args.code_base == expected


@pytest.mark.parametrize(
    'explicit,deepspeed,expected',
    [('', False, '/rocm10/Megatron-LM'), ('/explicit', False, '/explicit'), ('', True, '')]
)
def test_environment_path_available_during_dataset_generation(explicit, deepspeed, expected):
    """Apply the environment path before dataset generation and final path validation."""
    benchmark = make_benchmark('--mock_data --num_workers 0')
    benchmark._args.code_base = explicit
    benchmark._args.deepspeed = deepspeed
    with mock.patch.dict(os.environ, {'SB_MEGATRON_PATH': '/rocm10/Megatron-LM'}):
        assert benchmark._generate_dataset()
    assert benchmark._args.code_base == expected


@pytest.mark.parametrize('local_rank', [None, '0'])
def test_training_failure_restores_environment(local_rank):
    """Do not report failed subprocesses as successful timings or assign None to environ."""
    benchmark = make_benchmark()
    with mock.patch.dict(os.environ):
        os.environ.pop('OMPI_COMM_WORLD_LOCAL_RANK', None)
        if local_rank is not None:
            os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = local_rank
        with mock.patch(
            'superbench.benchmarks.model_benchmarks.megatron_gpt3.run_command',
            return_value=CompletedProcess('torchrun', 2, 'training failed')
        ):
            assert benchmark._train_step(Precision.FLOAT32) == ([], {})
        assert os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') == local_rank


def test_training_failure_fails_run():
    """A nonzero training exit must make run() return False, not only set a return code."""
    benchmark = MegatronGPT('megatron-gpt', '--precision float32 --model_action train')
    with mock.patch.object(MegatronGPT, '_preprocess', return_value=True), \
            mock.patch.object(MegatronGPT, '_postprocess', return_value=True), \
            mock.patch.object(MegatronGPT, '_megatron_command', return_value='torchrun pretrain_gpt.py'), \
            mock.patch.object(MegatronGPT, '_sync_result') as sync, \
            mock.patch(
                'superbench.benchmarks.model_benchmarks.megatron_gpt3.run_command',
                return_value=CompletedProcess('torchrun', 2, 'training failed')
            ):
        benchmark.add_parser_arguments()
        valid, benchmark._args, _ = benchmark.parse_args()
        assert valid
        benchmark._result = BenchmarkResult(
            benchmark._name, benchmark._benchmark_type, ReturnCode.SUCCESS, run_count=benchmark._args.run_count
        )
        assert benchmark.run() is False
    assert benchmark.return_code == ReturnCode.INVALID_BENCHMARK_RESULT
    sync.assert_not_called()
