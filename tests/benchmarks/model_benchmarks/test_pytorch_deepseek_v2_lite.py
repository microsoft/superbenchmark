# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for the DeepSeek-V2-Lite training benchmark: adapter, FSDP worker and MoE dispatch."""

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks import pytorch_deepseek_v2_lite as adapter
from superbench.benchmarks.model_benchmarks.deepseek_v2_lite import cache as cache_cleanup, moe, training_worker
from superbench.benchmarks.model_benchmarks.deepseek_v2_lite.utils import identity_report, write_json

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip('torch')

requires_python_310 = pytest.mark.skipif(
    sys.version_info < (3, 10), reason='The V2-Lite training runtime requires Python 3.10 or later.'
)


# Adapter.
@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    """Use an isolated cache root and a single local launcher."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.delenv('OMPI_COMM_WORLD_SIZE', raising=False)
    return adapter.DeepSeekV2LiteTraining(
        'deepseek-v2-lite-training',
        '--cache_root "{}" --num_steps 2 --seq_len 64 --duration 60'.format(tmp_path),
    )


def successful_worker(command, log, timeout, env):
    """Supply a worker-contract fixture, not a claim of model execution."""
    assert command[command.index('--module') + 1] == 'deepseek_v2_lite.training_worker'
    assert env['PYTHONPATH'].split(os.pathsep)[0] == str(Path(adapter.__file__).resolve().parent)
    output = Path(command[command.index('--output') + 1])
    dispatch = command[command.index('--moe_dispatch') + 1]
    report = dict(
        identity_report(),
        status='success',
        correctness_passed=True,
        teardown_complete=True,
        step_ms=[10.0, 20.0],
        loss=[2.0, 1.9],
        gradient_norm=[1.0, 0.9],
        loss_token_count=2 * 8 * 63,
        updated_parameter_elements=100,
        world_size=4,
        moe_dispatch=dispatch,
        moe_layers_rewritten=26 if dispatch == 'stable_sorted' else 0,
    )
    write_json(output, report)


def test_successful_adapter_metrics(benchmark, monkeypatch):
    """Preserve the MODEL result shape and count predicted loss tokens correctly."""
    monkeypatch.setattr(adapter, 'run_worker', successful_worker)
    assert benchmark.run()
    assert benchmark.type == BenchmarkType.MODEL
    assert benchmark.return_code == ReturnCode.SUCCESS
    assert benchmark.result['bf16_train_step_time'] == [15.0]
    assert benchmark.raw_data['bf16_train_step_time'] == [[10.0, 20.0]]
    assert benchmark.result['bf16_train_tokens_per_second'] == [2 * 8 * 63 * 1000 / 30]


@pytest.mark.parametrize(
    'parameters', [
        '--seq_len 0', '--num_gpus 1', '--batch_size 0', '--num_steps 0', '--num_warmup -1', '--learning_rate nan',
        '--duration 3600', '--run_count 2', '--hidden_size 1024', '--moe_dispatch unknown'
    ]
)
def test_invalid_profiles(tmp_path, parameters):
    """Reject invalid shapes, architecture overrides and combined timing profiles."""
    benchmark = adapter.DeepSeekV2LiteTraining(
        'deepseek-v2-lite-training', '--cache_root "{}" {}'.format(tmp_path, parameters)
    )
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT


def test_cache_root_is_required():
    """Refuse to choose an implicit location for the per-run model cache."""
    benchmark = adapter.DeepSeekV2LiteTraining('deepseek-v2-lite-training', '--num_steps 2')
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT


def test_parameters_render_in_help():
    """Keep every option compatible with the typed SuperBench help formatter."""
    benchmark = adapter.DeepSeekV2LiteTraining('deepseek-v2-lite-training')
    benchmark.add_parser_arguments()
    settings = benchmark.get_configurable_settings()
    assert '--moe_dispatch' in settings
    assert '--cache_root' in settings


@pytest.mark.parametrize('world_size', ['8', 'not-an-integer'])
def test_duplicate_launcher_is_rejected(benchmark, monkeypatch, world_size):
    """An MPI worker group must not launch eight independent full-model jobs."""
    monkeypatch.setenv('WORLD_SIZE', world_size)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT


def test_download_or_worker_failure_is_not_success(benchmark, monkeypatch):
    """Reject a failed subprocess rather than inventing throughput."""
    def failed_worker(command, log, timeout, env):
        raise RuntimeError('download failed')

    monkeypatch.setattr(adapter, 'run_worker', failed_worker)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.RUNTIME_EXCEPTION_ERROR
    assert 'bf16_train_step_time' not in benchmark.result


def test_timeout_is_failure(benchmark, monkeypatch):
    """Finishing through watchdog termination does not meet the time limit."""
    def timed_out_worker(command, log, timeout, env):
        raise TimeoutError('whole job deadline')

    monkeypatch.setattr(adapter, 'run_worker', timed_out_worker)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT


@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout'])
def test_model_cache_is_removed_after_every_run(benchmark, monkeypatch, outcome):
    """Delete the per-run checkpoint cache on every exit path while keeping reports and logs."""
    jobs = []

    def worker(command, log, timeout, env):
        cache = Path(command[command.index('--cache_dir') + 1])
        (cache / 'checkpoint').mkdir(parents=True)
        (cache / 'checkpoint' / 'model-00001-of-000004.safetensors').write_bytes(b'weights')
        log.write_text('worker log\n')
        jobs.append(cache.parent)
        if outcome == 'failure':
            raise RuntimeError('worker failed')
        if outcome == 'timeout':
            raise TimeoutError('whole job deadline')
        successful_worker(command, log, timeout, env)

    monkeypatch.setattr(adapter, 'run_worker', worker)
    assert benchmark.run() is (outcome == 'success')
    assert not (jobs[0] / 'cache').exists()
    assert (jobs[0] / 'worker.log').read_text() == 'worker log\n'
    assert (jobs[0] / 'report.json').is_file() is (outcome == 'success')


@pytest.mark.parametrize('cleanup_seconds', [7, 50, 70])
def test_cache_cleanup_counts_toward_the_job_deadline(benchmark, monkeypatch, cleanup_seconds):
    """Include cleanup in elapsed time and reject the exact deadline as well as an overrun."""
    clock = [100.0]
    run_cleanup = subprocess.run

    def worker(command, log, timeout, env):
        assert timeout == 40
        Path(command[command.index('--cache_dir') + 1]).mkdir(parents=True)
        successful_worker(command, log, timeout, env)
        clock[0] += 10

    def cleanup(command, **kwargs):
        assert kwargs['timeout'] == 50
        result = run_cleanup(command, **kwargs)
        clock[0] += cleanup_seconds
        return result

    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(adapter, 'time', fake_time)
    monkeypatch.setattr(cache_cleanup, 'time', fake_time)
    monkeypatch.setattr(adapter, 'run_worker', worker)
    monkeypatch.setattr(cache_cleanup.subprocess, 'run', cleanup)
    assert benchmark.run() is (cleanup_seconds < 50)
    if cleanup_seconds < 50:
        assert benchmark.return_code == ReturnCode.SUCCESS
        assert benchmark.result['end_to_end_seconds'] == [10 + cleanup_seconds]
        assert benchmark.result['bf16_train_tokens_per_second'] == [2 * 8 * 63 * 1000 / 30]
    else:
        assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT
        assert 'end_to_end_seconds' not in benchmark.result
        assert 'bf16_train_tokens_per_second' not in benchmark.result


@pytest.mark.parametrize(
    'failure,code', [
        (RuntimeError('worker failed'), ReturnCode.RUNTIME_EXCEPTION_ERROR),
        (TimeoutError('whole job deadline'), ReturnCode.KILLED_BY_TIMEOUT),
    ]
)
def test_unfinished_failure_cleanup_keeps_the_original_error(benchmark, monkeypatch, failure, code):
    """Report the worker failure rather than a later cleanup timeout, and log both."""
    log_error = Mock()

    def worker(command, log, timeout, env):
        Path(command[command.index('--cache_dir') + 1]).mkdir(parents=True)
        raise failure

    monkeypatch.setattr(adapter, 'run_worker', worker)
    monkeypatch.setattr(cache_cleanup.subprocess, 'run', Mock(side_effect=subprocess.TimeoutExpired(['cleanup'], 1)))
    monkeypatch.setattr(adapter.logger, 'error', log_error)
    assert not benchmark.run()
    assert benchmark.return_code == code
    messages = [call[0][0] % call[0][1:] for call in log_error.call_args_list]
    assert any(str(failure) in message for message in messages)
    assert any('after a failed run' in message for message in messages)


@pytest.mark.parametrize(
    'field,value', [
        ('step_ms', [-1, 20]), ('step_ms', [10]), ('correctness_passed', False), ('teardown_complete', False),
        ('updated_parameter_elements', True), ('loss_token_count', 1024), ('moe_dispatch', None),
        ('moe_dispatch', 'stable_sorted'), ('moe_layers_rewritten', False), ('moe_layers_rewritten', 26)
    ]
)
def test_invalid_worker_report(benchmark, monkeypatch, field, value):
    """Check required metric thresholds and shapes, not only process exit zero."""
    def invalid_worker(command, log, timeout, env):
        successful_worker(command, log, timeout, env)
        output = Path(command[command.index('--output') + 1])
        report = adapter.read_json(output)
        report[field] = value
        write_json(output, report)

    monkeypatch.setattr(adapter, 'run_worker', invalid_worker)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.RUNTIME_EXCEPTION_ERROR


def test_python_and_path_remain_separate_arguments(benchmark):
    """Never concatenate runtime paths or forward unchecked model flags."""
    assert benchmark._preprocess()
    benchmark._args.runtime_python = '/path with spaces/python'
    command = benchmark._worker_command(Path('/cache with spaces/job'))
    assert command[0] == '/path with spaces/python'
    assert command[command.index('--cache_dir') + 1] == '/cache with spaces/job/cache'
    assert '--nproc_per_node=4' in command
    assert command[command.index('--module') + 1] == 'deepseek_v2_lite.training_worker'
    assert command[command.index('--moe_dispatch') + 1] == 'official'


def test_opt_in_dispatch_reaches_the_worker_and_requires_matching_evidence(benchmark, tmp_path, monkeypatch):
    """Thread the explicit dispatch selection through the real adapter contract."""
    selected = adapter.DeepSeekV2LiteTraining(
        'deepseek-v2-lite-training',
        '--cache_root "{}" --num_steps 2 --seq_len 64 --duration 60 --moe_dispatch stable_sorted'.format(tmp_path),
    )
    commands = []

    def worker(command, log, timeout, env):
        commands.append(command)
        successful_worker(command, log, timeout, env)

    monkeypatch.setattr(adapter, 'run_worker', worker)
    assert selected.run()
    assert commands[0][commands[0].index('--moe_dispatch') + 1] == 'stable_sorted'


# FSDP worker.
@pytest.fixture
def worker(monkeypatch):
    """Return the worker and replace only GPU/distributed synchronization in CPU tests."""
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda tensor: None)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    tensor = torch.tensor

    def cpu_tensor(data, *args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    return training_worker


class UpdateFixture(torch.nn.Module):
    """Use a small real autograd/optimizer update, not a full-model proxy result."""
    def __init__(self):
        """Initialize a nonzero parameter and loss."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, input_ids, labels, use_cache):
        """Return a scalar differentiable loss without external model dependencies."""
        return SimpleNamespace(loss=self.weight.sum().square())

    def clip_grad_norm_(self, maximum):
        """Expose the FSDP clipping interface using the CPU parameters."""
        return torch.nn.utils.clip_grad_norm_(self.parameters(), maximum)


def test_host_snapshot_still_detects_real_update(worker):
    """Host snapshots must be independent copies even for the CPU test model."""
    model = UpdateFixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    elapsed, loss, norm, changed = worker.training_step(model, optimizer, torch.tensor([[1, 2]]), True)
    assert elapsed > 0 and loss > 0 and norm > 0
    assert changed == 4
    assert torch.all(model.weight < 1).item()


def test_optimizer_noop_is_not_success(worker, monkeypatch):
    """Moving validation storage to host must not weaken the actual-update gate."""
    model = UpdateFixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    monkeypatch.setattr(optimizer, 'step', lambda: None)
    with pytest.raises(RuntimeError, match='did not update'):
        worker.training_step(model, optimizer, torch.tensor([[1, 2]]), True)


@pytest.mark.parametrize('invalid', ['missing_loss', 'nonfinite_loss', 'zero_gradient'])
def test_training_step_rejects_invalid_numerics(worker, monkeypatch, invalid):
    """Never publish timing for a missing loss, NaN or zero-gradient workload."""
    model = UpdateFixture()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    if invalid == 'missing_loss':
        monkeypatch.setattr(model, 'forward', lambda **kwargs: SimpleNamespace(loss=None))
    elif invalid == 'nonfinite_loss':
        monkeypatch.setattr(model, 'forward', lambda **kwargs: SimpleNamespace(loss=model.weight.sum() * float('nan')))
    else:
        monkeypatch.setattr(model, 'clip_grad_norm_', lambda maximum: torch.tensor(0.0))
    with pytest.raises(RuntimeError):
        worker.training_step(model, optimizer, torch.tensor([[1, 2]]))


@pytest.fixture
def main_driver(worker, tmp_path, monkeypatch):
    """Exercise worker orchestration with explicit fake GPU/model boundaries."""
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '0')
    monkeypatch.setattr(worker.signal, 'signal', Mock())
    monkeypatch.setattr(worker, 'set_cache_environment', Mock())
    monkeypatch.setattr(worker, 'configure_cache', Mock())
    download = Mock(return_value=tmp_path / 'checkpoint')
    monkeypatch.setattr(worker, 'download_checkpoint', download)
    model = Mock()
    model.cuda.return_value = model
    loader = Mock(return_value=model)
    reference = Mock(return_value={'world_size': 2, 'batch_size': 2, 'seq_len': 64, 'reference_microbatch_size': 2})
    training = Mock(return_value={'world_size': 2, 'step_ms': [1.0, 2.0]})
    monkeypatch.setattr(worker, 'load_model', loader)
    monkeypatch.setattr(worker, 'training_reference', reference)
    monkeypatch.setattr(worker, 'run_training', training)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'empty_cache', Mock())
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'CPU orchestration fixture')
    monkeypatch.setattr(torch, 'manual_seed', Mock())
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    initialize = Mock()
    control = object()
    destroy = Mock()
    monkeypatch.setattr(torch.distributed, 'init_process_group', initialize)
    monkeypatch.setattr(torch.distributed, 'new_group', Mock(return_value=control))
    monkeypatch.setattr(torch.distributed, 'barrier', Mock())
    monkeypatch.setattr(torch.distributed, 'destroy_process_group', destroy)
    output = tmp_path / 'report.json'
    argv = [
        'worker',
        '--cache_dir',
        str(tmp_path / 'cache'),
        '--output',
        str(output),
        '--num_gpus',
        '2',
        '--batch_size',
        '2',
        '--seq_len',
        '64',
        '--num_steps',
        '2',
        '--duration',
        '60',
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    return SimpleNamespace(
        output=output,
        argv=argv,
        download=download,
        loader=loader,
        reference=reference,
        training=training,
        initialize=initialize,
        destroy=destroy,
        control=control,
    )


def test_worker_main_preserves_profile_and_reaps_groups(worker, main_driver):
    """A success report follows reference/training completion and both group teardowns."""
    state = main_driver
    worker.main()
    report = json.loads(state.output.read_text())
    assert report['status'] == 'success' and report['correctness_passed'] and report['teardown_complete']
    assert state.loader.call_args[0][1] == torch.bfloat16
    assert state.reference.call_args[0][2:] == (2, 64, 2)
    assert [call[0] for call in state.destroy.call_args_list] == [(state.control, ), ()]
    assert state.training.call_args[0][0].seq_len == 64
    assert state.training.call_args[0][0].moe_dispatch == 'official'
    assert 0 < report['total_seconds'] < 60


def test_worker_main_forwards_the_explicit_dispatch_selection(worker, main_driver, monkeypatch):
    """Select stable dispatch only for measured training, retaining the official reference loader."""
    monkeypatch.setattr(sys, 'argv', main_driver.argv + ['--moe_dispatch', 'stable_sorted'])
    worker.main()
    assert main_driver.training.call_args[0][0].moe_dispatch == 'stable_sorted'
    assert main_driver.loader.call_args[0][1] == torch.bfloat16


def test_nonzero_rank_does_not_download_or_publish(worker, main_driver, monkeypatch):
    """Only rank zero owns checkpoint/reference creation and the final report."""
    state = main_driver
    monkeypatch.setenv('RANK', '1')
    monkeypatch.setenv('LOCAL_RANK', '1')
    state.output.with_name('training-reference.json').write_text(json.dumps({'world_size': 2}))
    worker.main()
    state.download.assert_not_called()
    state.loader.assert_not_called()
    assert not state.output.exists()
    assert state.destroy.call_count == 2


def test_worker_failure_still_destroys_groups(worker, main_driver):
    """A failed numerical/training dependency cannot produce success or leak groups."""
    main_driver.training.side_effect = RuntimeError('numerical mismatch')
    with pytest.raises(RuntimeError, match='numerical mismatch'):
        worker.main()
    assert not main_driver.output.exists()
    assert main_driver.destroy.call_count == 2


@pytest.mark.parametrize(
    'override', [
        ['--num_gpus', '1'],
        ['--batch_size', '0'],
        ['--seq_len', '16'],
        ['--num_warmup', '-1'],
        ['--num_steps', '0'],
        ['--duration', '3600'],
        ['--learning_rate', 'nan'],
    ]
)
def test_worker_rejects_invalid_profile_before_initializing(worker, main_driver, monkeypatch, override):
    """Bad profiles must not start collective groups or download weights."""
    monkeypatch.setattr(sys, 'argv', main_driver.argv + override)
    with pytest.raises(ValueError):
        worker.main()
    main_driver.initialize.assert_not_called()
    main_driver.download.assert_not_called()


@pytest.mark.parametrize('checkpointed', [False, True])
def test_gradient_probe_mapping_covers_root_and_checkpointed_layers(worker, monkeypatch, checkpointed):
    """Exercise actual parameter lookup while replacing only FSDP's collective context."""
    fsdp = pytest.importorskip('torch.distributed.fsdp').FullyShardedDataParallel
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.embed_tokens = torch.nn.Embedding(2, 2)
    layer = torch.nn.Linear(2, 2, bias=False)
    for parameter in [root.model.embed_tokens.weight, layer.weight]:
        parameter.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    module = checkpoint_wrapper(layer) if checkpointed else layer
    root.model.layers = [SimpleNamespace(module=module)]
    model = SimpleNamespace(module=root)
    monkeypatch.setattr(fsdp, 'summon_full_params', lambda *args, **kwargs: nullcontext())
    reference = {
        'gradients': {
            'model.embed_tokens.weight': {
                'shape': [2, 2],
                'indices': [0, 3],
                'values': [1.0, 4.0]
            },
            'model.layers.0.weight': {
                'shape': [2, 2],
                'indices': [0, 3],
                'values': [1.0, 4.0]
            },
        }
    }
    assert worker.compare_gradients(model, reference) == 0
    reference['gradients']['model.layers.0.weight']['values'][0] = 10.0
    with pytest.raises(AssertionError):
        worker.compare_gradients(model, reference)
    layer.weight.grad = None
    with pytest.raises(RuntimeError, match='gradient'):
        worker.compare_gradients(model, reference)


# Stable MoE dispatch.
class Auxiliary(torch.autograd.Function):
    """Match the official auxiliary-loss autograd contract."""
    @staticmethod
    def forward(ctx, output, loss):
        """Return routed activations without changing the forward loss."""
        return output

    @staticmethod
    def backward(ctx, gradient):
        """Preserve activation gradients and differentiate the auxiliary loss."""
        return gradient, gradient.new_ones(())


AddAuxiliaryLoss = Auxiliary


class Gate(torch.nn.Module):
    """Provide deterministic routing with an optionally unused expert."""
    def __init__(self, experts, hidden, empty_expert):
        """Create trainable router weights for the control."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(experts, hidden))
        self.empty_expert = empty_expert

    def forward(self, values):
        """Return top-two routes and a differentiable auxiliary loss."""
        scores = torch.nn.functional.linear(values.reshape(-1, values.shape[-1]), self.weight).softmax(-1)
        routed = scores[:, :-1] if self.empty_expert else scores
        indices = routed.topk(2, dim=-1).indices
        return indices, scores.gather(1, indices), scores.square().mean()


class MoE(torch.nn.Module):
    """Implement the original Boolean-indexed reference dispatch."""
    def __init__(self, empty_expert, shared):
        """Create expert, router and optional shared-expert parameters."""
        super().__init__()
        self.num_experts_per_tok = 2
        self.config = SimpleNamespace(n_shared_experts=1 if shared else None)
        self.experts = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.SiLU(), torch.nn.Linear(6, 4)) for _ in range(4)]
        )
        self.gate = Gate(4, 4, empty_expert)
        if shared:
            self.shared_experts = torch.nn.Linear(4, 4)

    def forward(self, hidden):
        """Evaluate every expert in the original route order."""
        shape = hidden.shape
        ids, weights, auxiliary = self.gate(hidden)
        values = hidden.reshape(-1, shape[-1]).repeat_interleave(2, dim=0)
        result = torch.empty_like(values)
        for index, expert in enumerate(self.experts):
            selected = ids.reshape(-1) == index
            result[selected] = expert(values[selected])
        result = (result.view(*weights.shape, -1) * weights.unsqueeze(-1)).sum(1).view(shape)
        result = Auxiliary.apply(result, auxiliary)
        return result + self.shared_experts(hidden) if self.config.n_shared_experts else result


class DeepseekV2MoE(MoE):
    """Use the pinned official class contract for installation controls."""

    ep_size = 1


@requires_python_310
@pytest.mark.parametrize('empty_expert', [False, True])
@pytest.mark.parametrize('shared', [False, True])
def test_sorted_dispatch_preserves_outputs_inputs_parameters_and_auxiliary_gradients(empty_expert, shared):
    """Retain outputs and all gradients, including empty and shared experts."""
    torch.manual_seed(1234)
    model = MoE(empty_expert, shared).double()
    values = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    original = model(values)
    original.square().mean().backward()
    expected_input = values.grad.clone()
    expected_parameters = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    expected_output = original.detach().clone()
    model.zero_grad(set_to_none=True)
    values.grad = None
    actual = moe.sorted_training_forward(model, values, Auxiliary.apply)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected_output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(values.grad, expected_input, rtol=1e-12, atol=1e-12)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected_parameters[name], rtol=1e-12, atol=1e-12)
    if empty_expert:
        assert torch.count_nonzero(model.experts[-1][0].weight.grad).item() == 0


@requires_python_310
def test_installation_preserves_parameters_and_the_evaluation_path(monkeypatch):
    """Only training dispatch changes; evaluation retains the original forward."""
    model = torch.nn.ModuleList([DeepseekV2MoE(False, True).double() for _ in range(26)])
    parameters = [id(parameter) for parameter in model.parameters()]
    values = torch.randn(1, 2, 4, dtype=torch.float64)
    model.eval()
    expected = model[0](values).detach()
    assert moe.install_sorted_dispatch(model) == 26
    assert [id(parameter) for parameter in model.parameters()] == parameters
    replacement = Mock(side_effect=RuntimeError('training dispatch reached'))
    monkeypatch.setattr(moe, 'sorted_training_forward', replacement)
    torch.testing.assert_close(model[0](values), expected)
    replacement.assert_not_called()
    model.train()
    with pytest.raises(RuntimeError, match='training dispatch reached'):
        model[0](values)


@requires_python_310
@pytest.mark.parametrize('count,expert_parallelism', [(25, 1), (27, 1), (26, 2)])
def test_unsupported_layout_is_rejected_before_any_forward_changes(count, expert_parallelism):
    """A partial model or expert-parallel layout cannot receive a partial patch."""
    model = torch.nn.ModuleList([DeepseekV2MoE(False, False) for _ in range(count)])
    model[0].ep_size = expert_parallelism
    original = [module.forward for module in model]
    with pytest.raises(ValueError):
        moe.install_sorted_dispatch(model)
    assert [module.forward for module in model] == original
