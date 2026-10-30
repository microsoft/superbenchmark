# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for the shared DeepSeek-V2-Lite helpers: checkpoint, worker supervision, cache and reference."""

import copy
import ctypes
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from superbench.benchmarks.model_benchmarks.deepseek_v2_lite import (
    cache as cleanup, reference as reference_module, utils
)

torch = pytest.importorskip('torch')


# Checkpoint identity, fixed inputs and worker supervision.
def test_checkpoint_identity():
    """Reject architecture proxies and quantization."""
    config = dict(utils.EXPECTED_CONFIG, model_type='deepseek_v2')
    utils.validate_config(config)
    config['hidden_size'] = 1024
    with pytest.raises(ValueError, match='hidden_size'):
        utils.validate_config(config)
    config['hidden_size'] = 2048
    config['quantization_config'] = {'quant_method': 'fp8'}
    with pytest.raises(ValueError, match='quantized'):
        utils.validate_config(config)
    assert utils.WEIGHT_BYTES == 31413626576


def test_packed_text_is_reproducible():
    """Pack actual token inputs with EOS separators, not padding or mock execution."""
    documents = [[1, 2], [3, 4, 5]]
    expected = [[1, 2, 9, 3], [4, 5, 9, 1]]
    assert utils.packed_tokens(documents, 9, 4, 2) == expected
    assert utils.packed_tokens(documents, 9, 4, 2) == expected
    assert utils.packed_tokens(documents, 9, 4, 2, offset=1) != expected


@pytest.mark.parametrize('documents,length,count', [([], 4, 1), ([[]], 4, 1), ([[True]], 4, 1), ([[1]], 1, 1)])
def test_invalid_text_inputs(documents, length, count):
    """Surface malformed inputs instead of returning an empty batch."""
    with pytest.raises(ValueError):
        utils.packed_tokens(documents, 9, length, count)


def test_cold_cache_isolated(tmp_path, monkeypatch):
    """Reject cache reuse without deleting shared contents."""
    shared = tmp_path / 'shared'
    shared.mkdir()
    sentinel = shared / 'keep'
    sentinel.write_text('unrelated data')
    monkeypatch.setenv('HF_HOME', str(shared))
    cache = tmp_path / 'cold'
    utils.configure_cache(cache)
    assert sentinel.read_text() == 'unrelated data'
    assert utils.os.environ['HF_HOME'] == str(cache / 'hf')
    assert utils.os.environ['HF_XET_CACHE'] == str(cache / 'xet')
    (cache / 'present').touch()
    with pytest.raises(ValueError, match='empty'):
        utils.configure_cache(cache)


def test_report_requires_correctness_and_teardown():
    """Require model identity, not just a successful process exit."""
    report = dict(utils.identity_report(), status='success', correctness_passed=True, teardown_complete=True)
    utils.validate_identity(report)
    incorrect = copy.deepcopy(report)
    incorrect['parameter_count'] = 100
    with pytest.raises(ValueError, match='contract'):
        utils.validate_identity(incorrect)
    report['correctness_passed'] = False
    with pytest.raises(ValueError, match='Correctness'):
        utils.validate_identity(report)


@pytest.mark.parametrize('length', [1, 512, 209715200, 15706484224])
def test_large_tensor_probe_indices(length):
    """FP32-rounded endpoints must never index beyond a real full-model tensor."""
    indices = utils.uniform_indices(length)
    assert len(indices) == 256
    assert indices[0] == 0 and indices[-1] == length - 1
    assert all(0 <= index < length for index in indices)


def test_atomic_json_rejects_nonfinite_values(tmp_path):
    """A malformed report must not replace a complete report."""
    path = tmp_path / 'report.json'
    utils.write_json(path, {'value': 1})
    with pytest.raises(ValueError):
        utils.write_json(path, {'value': math.nan})
    assert utils.read_json(path) == {'value': 1}
    path.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match='Nonfinite'):
        utils.read_json(path)


def test_worker_failure_is_explicit(tmp_path):
    """Reject nonzero exit codes even if the worker writes a result-like message."""
    with pytest.raises(RuntimeError, match='code 7'):
        utils.run_worker([sys.executable, '-c', 'print("success"); raise SystemExit(7)'], tmp_path / 'worker.log', 5)
    assert 'success' in (tmp_path / 'worker.log').read_text()


def test_timeout_reaps_worker(tmp_path):
    """A timeout must terminate the created process rather than leave a worker."""
    pid_file = tmp_path / 'pid'
    command = [
        sys.executable,
        '-c',
        'import os,time; from pathlib import Path; Path({!r}).write_text(str(os.getpid())); time.sleep(60)'.format(
            str(pid_file)
        ),
    ]
    with pytest.raises(TimeoutError, match='deadline'):
        utils.run_worker(command, tmp_path / 'worker.log', 1)
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        utils.os.kill(pid, 0)


def test_success_with_live_child_is_rejected(tmp_path):
    """A leader's exit code zero is insufficient when its child remains alive."""
    command = [
        sys.executable,
        '-c',
        'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])',
    ]
    with pytest.raises(RuntimeError, match='descendant processes'):
        utils.run_worker(command, tmp_path / 'worker.log', 5)


@pytest.mark.parametrize('launcher_exits', [False, True])
def test_separate_session_child_is_reaped(tmp_path, launcher_exits):
    """Own escaped rank sessions even when their launcher exits or ignores termination."""
    pid_file = tmp_path / 'rank.pid'
    rank = (
        'import os,signal,time; from pathlib import Path; '
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        'Path({!r}).write_text(str(os.getpid())); time.sleep(60)'
    ).format(str(pid_file))
    launcher = (
        'import signal,subprocess,sys,time; from pathlib import Path; '
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        'subprocess.Popen([sys.executable, "-c", {!r}], start_new_session=True); '
        'ready=Path({!r})\n'
        'while not ready.exists(): time.sleep(0.01)\n'
        '{}'
    ).format(rank, str(pid_file), 'sys.exit(0)' if launcher_exits else 'time.sleep(60)')
    try:
        expected = RuntimeError if launcher_exits else TimeoutError
        with pytest.raises(expected):
            utils.run_worker([sys.executable, '-c', launcher], tmp_path / 'worker.log', 5 if launcher_exits else 1)
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if pid_file.exists():
            pid = int(pid_file.read_text())
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_worker_signal_does_not_kill_unrelated_process(tmp_path):
    """Timeout cleanup must be scoped to the supervised job, not the controller's children."""
    unrelated = utils.subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        with pytest.raises(TimeoutError):
            utils.run_worker([sys.executable, '-c', 'import time; time.sleep(60)'], tmp_path / 'worker.log', 0.2)
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_controller_death_reaps_owned_worker(tmp_path):
    """A killed controller must not leave its independently supervised job alive."""
    pid_file = tmp_path / 'worker.pid'
    worker = ('import os,time; from pathlib import Path; '
              'Path({!r}).write_text(str(os.getpid())); time.sleep(60)').format(str(pid_file))
    controller = (
        'import sys; from pathlib import Path; sys.path.insert(0, {!r}); '
        'from deepseek_v2_lite.utils import run_worker; '
        'run_worker([sys.executable, "-c", {!r}], Path({!r}), 60)'
    ).format(str(Path(utils.__file__).resolve().parents[1]), worker, str(tmp_path / 'worker.log'))
    process = utils.subprocess.Popen([sys.executable, '-c', controller])
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists()
        pid = int(pid_file.read_text())
        process.kill()
        process.wait(timeout=5)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            pytest.fail('The worker survived its controller.')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def checkpoint_files(tmp_path, monkeypatch):
    """Create small real files to test the checkpoint integrity contract."""
    root = tmp_path / 'checkpoint'
    root.mkdir()
    contents = b'CPU checkpoint integrity fixture'
    shard = root / 'weights.safetensors'
    shard.write_bytes(contents)
    monkeypatch.setattr(utils, 'WEIGHT_FILES', {
        shard.name: (len(contents), hashlib.sha256(contents).hexdigest()),
    })
    utils.write_json(root / 'config.json', dict(utils.EXPECTED_CONFIG, model_type='deepseek_v2'))
    utils.write_json(root / 'model.safetensors.index.json', {'weight_map': {'fixture.weight': shard.name}})
    return root, shard


@pytest.mark.parametrize('failure', ['missing', 'size', 'hash', 'index'])
def test_invalid_checkpoint_files_fail(checkpoint_files, failure):
    """Corrupted, missing or misindexed shards must never become a valid checkpoint."""
    root, shard = checkpoint_files
    if failure == 'missing':
        shard.unlink()
    elif failure == 'size':
        shard.write_bytes(b'short')
    elif failure == 'hash':
        shard.write_bytes(b'x' * shard.stat().st_size)
    else:
        utils.write_json(root / 'model.safetensors.index.json', {'weight_map': {'fixture.weight': 'other'}})
    with pytest.raises(ValueError):
        utils.verify_checkpoint(root)


def test_download_uses_pinned_public_identity_and_then_verifies(checkpoint_files, monkeypatch):
    """Disable implicit credentials and require the downloaded shard contract."""
    root, shard = checkpoint_files
    download = Mock()
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=download))
    assert utils.download_checkpoint(root.parent) == root
    arguments = download.call_args[1]
    assert arguments['repo_id'] == utils.MODEL_ID
    assert arguments['revision'] == utils.REVISION
    assert arguments['token'] is False
    assert arguments['local_dir'] == str(root)
    assert shard.name in arguments['allow_patterns']


@pytest.mark.parametrize('timeout', [0, -1, float('nan')])
def test_invalid_worker_deadline_fails_before_launch(tmp_path, timeout):
    """A nonfinite or expired budget must not start a worker."""
    with pytest.raises(TimeoutError):
        utils.run_worker([sys.executable, '-c', 'pass'], tmp_path / 'worker.log', timeout)


def test_worker_interrupt_is_failure():
    """A termination request must unwind the worker's normal success path."""
    with pytest.raises(TimeoutError, match='interrupted'):
        utils.interrupt_worker(signal.SIGTERM, None)


def test_supervisor_requires_kernel_ownership_support(monkeypatch):
    """Fail before spawning when kernel ownership setup is unavailable."""
    monkeypatch.setattr(utils.signal, 'signal', Mock())
    monkeypatch.setattr(ctypes, 'CDLL', lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: -1))
    monkeypatch.setattr(ctypes, 'get_errno', lambda: 13)
    spawn = Mock()
    monkeypatch.setattr(utils.subprocess, 'Popen', spawn)
    with pytest.raises(OSError):
        utils.supervise_worker([sys.executable, '-c', 'pass'], 1, os.getppid())
    spawn.assert_not_called()


# Per-run model cache removal.
def test_removes_only_the_run_cache(tmp_path):
    """Delete the downloaded checkpoint while keeping the run's report and logs."""
    cache = tmp_path / 'cache'
    (cache / 'checkpoint').mkdir(parents=True)
    (cache / 'checkpoint' / 'model-00001-of-000004.safetensors').write_bytes(b'weights')
    (tmp_path / 'report.json').write_text('{}')
    cleanup.remove_model_cache(cache, time.monotonic() + 30)
    assert not cache.exists()
    assert (tmp_path / 'report.json').is_file()


def test_cleanup_does_not_start_without_budget(tmp_path, monkeypatch):
    """Never start an unbounded deletion after the deadline."""
    run = Mock()
    monkeypatch.setattr(cleanup.subprocess, 'run', run)
    with pytest.raises(TimeoutError, match='No time remains'):
        cleanup.remove_model_cache(tmp_path / 'cache', time.monotonic())
    run.assert_not_called()


@pytest.mark.parametrize(
    'failure',
    [PermissionError('read-only cache'),
     subprocess.CalledProcessError(1, ['cleanup'], output='read-only cache')]
)
def test_filesystem_errors_are_logged(tmp_path, monkeypatch, failure):
    """Report a cleanup error without treating it as a failed measurement."""
    log_error = Mock()
    monkeypatch.setattr(cleanup.subprocess, 'run', Mock(side_effect=failure))
    monkeypatch.setattr(cleanup.logger, 'error', log_error)
    cleanup.remove_model_cache(tmp_path / 'cache', time.monotonic() + 30)
    log_error.assert_called_once()
    assert 'read-only cache' in str(log_error.call_args[0][-1])


@pytest.mark.parametrize('interrupted', [False, True])
def test_cleanup_timeout_reaps_its_process(tmp_path, monkeypatch, interrupted):
    """Stop and reap a real stalled cleanup child on either its deadline or controller interruption."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    children = []
    popen = subprocess.Popen

    def stalled_cleanup(command, **kwargs):
        child = popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        if interrupted:
            monkeypatch.setattr(child, 'communicate', Mock(side_effect=TimeoutError('controller interrupted')))
        return child

    monkeypatch.setattr(cleanup.subprocess, 'Popen', stalled_cleanup)
    start = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            cleanup.remove_model_cache(cache, start + 0.25)
        assert time.monotonic() - start < 3
        assert len(children) == 1
        assert children[0].returncode is not None
        assert cache.is_dir()
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


@pytest.mark.parametrize('deadline,budget', [(400.0, cleanup.FAILURE_CLEANUP_GRACE_SECONDS), (1000.0, 500.0)])
def test_failure_cleanup_uses_the_remaining_budget_or_grace(tmp_path, monkeypatch, deadline, budget):
    """Give a failed run's cleanup the rest of its budget, or a fixed grace once that budget is spent."""
    run = Mock()
    monkeypatch.setattr(cleanup, 'time', SimpleNamespace(monotonic=lambda: 500.0))
    monkeypatch.setattr(cleanup.subprocess, 'run', run)
    cleanup.remove_model_cache_after_failure(tmp_path / 'cache', deadline)
    assert run.call_args[1]['timeout'] == budget


@pytest.mark.parametrize(
    'failure', [subprocess.TimeoutExpired(['cleanup'], 1),
                TimeoutError('Killed by SIGTERM or timeout!')]
)
def test_failure_cleanup_never_raises_a_timeout(tmp_path, monkeypatch, failure):
    """Log an unfinished cleanup so that the failed run keeps its original error."""
    log_error = Mock()
    monkeypatch.setattr(cleanup.subprocess, 'run', Mock(side_effect=failure))
    monkeypatch.setattr(cleanup.logger, 'error', log_error)
    cleanup.remove_model_cache_after_failure(tmp_path / 'cache', time.monotonic())
    log_error.assert_called_once()
    assert 'after a failed run' in log_error.call_args[0][0]


# Official-code reference.
@pytest.fixture
def reference(monkeypatch):
    """Return the reference and redirect only test tensor creation to CPU."""
    tensor = torch.tensor

    def cpu_tensor(data, *args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    return reference_module


class ReferenceFixture:
    """Expose token-independent logits while checking cached attention shapes."""
    def __init__(self):
        """Track cache and full-prefix calls separately."""
        self.calls = []

    def eval(self):
        """Retain the minimal evaluation interface used by the oracle."""
        return self

    def __call__(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
        """Require a mask for all past and current positions, including one-token decode."""
        length = input_ids.shape[1]
        total = length + (past_key_values or 0)
        assert attention_mask.shape == (1, total)
        assert attention_mask.all().item()
        self.calls.append((length, total, past_key_values, use_cache))
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 1, 4).repeat(1, length, 1)
        return SimpleNamespace(logits=logits, past_key_values=total if use_cache else None)


def request_fixture():
    """Provide exact fixture probabilities rather than a model-quality claim."""
    probabilities = torch.tensor([0.0, 1.0, 2.0, 3.0]).log_softmax(-1).tolist()
    return {
        'prompt_token_ids': [1, 2],
        'token_ids': [3, 3],
        'logprobs': [{str(index): value
                      for index, value in enumerate(probabilities)} for _ in range(2)],
    }


def test_cached_decode_mask_covers_full_prefix(reference):
    """The second cache call has one input token but a three-position mask."""
    model = ReferenceFixture()
    result = reference.inference_reference(model, [request_fixture()])
    assert model.calls == [(2, 2, None, True), (2, 2, None, False), (1, 3, 2, True), (3, 3, None, False)]
    assert len(result['comparisons']) == 2
    assert all(row['cache_max_abs_error'] == 0 for row in result['comparisons'])


@pytest.mark.parametrize('requests', [[], [{'prompt_token_ids': [], 'token_ids': [], 'logprobs': []}]])
def test_incomplete_reference_requests_fail(reference, requests):
    """Never accept an empty or truncated numerical workload."""
    with pytest.raises(ValueError, match='nonempty|Incomplete'):
        reference.inference_reference(ReferenceFixture(), requests)


def test_reference_requires_greedy_token_agreement(reference):
    """Finite probabilities alone must not certify an incorrect generated token."""
    request = request_fixture()
    request['token_ids'][0] = 1
    with pytest.raises(RuntimeError, match='greedy token'):
        reference.inference_reference(ReferenceFixture(), [request])


def test_reference_does_not_ignore_probability_mismatch(reference):
    """Keep the numerical gate even when all generated tokens are correct."""
    request = request_fixture()
    request['logprobs'][0]['3'] += 1.0
    with pytest.raises(AssertionError, match='request 0 step 0'):
        reference.inference_reference(ReferenceFixture(), [request])


@pytest.mark.parametrize('positions', [[], [0], [1], [0, 0, 1], [1, 0], [0, 2], [False, 1]])
def test_invalid_full_prefix_sampling_fails(reference, positions):
    """Do not accept malformed or first/final-step-incomplete numerical evidence."""
    request = request_fixture()
    request['full_prefix_steps'] = positions
    with pytest.raises(ValueError, match='reference positions'):
        reference.inference_reference(ReferenceFixture(), [request])


class TrainingReferenceFixture:
    """Small autograd fixture for global-batch partitioning, not GPU model validation."""
    def __init__(self):
        """Create the named parameters required by reference sampling."""
        names = [
            'model.embed_tokens.weight',
            'model.layers.0.self_attn.q_proj.weight',
            'model.layers.0.self_attn.kv_a_proj_with_mqa.weight',
            'model.layers.0.mlp.down_proj.weight',
            'model.layers.1.mlp.gate.weight',
            'model.layers.1.mlp.experts.0.down_proj.weight',
        ]
        self.parameters = {name: torch.nn.Parameter(torch.ones(2, 2)) for name in names}
        self.batches = []
        self.checkpointing = False

    def train(self):
        """Expose the reference model's training-mode interface."""
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        """Require the memory-bounded nonreentrant reference policy."""
        assert gradient_checkpointing_kwargs == {'use_reentrant': False}
        self.checkpointing = True

    def named_parameters(self):
        """Expose stable names for independently selected gradient probes."""
        return self.parameters.items()

    def get_parameter(self, name):
        """Look up the same parameter on each logical rank batch."""
        return self.parameters[name]

    def zero_grad(self, set_to_none):
        """Discard local gradients before the next reference batch."""
        assert set_to_none
        for parameter in self.parameters.values():
            parameter.grad = None

    def __call__(self, input_ids, labels, use_cache):
        """Produce input-dependent gradients that differ between rank batches."""
        assert not use_cache and torch.equal(input_ids, labels)
        self.batches.append(input_ids.clone())
        loss = sum(parameter.sum() for parameter in self.parameters.values()) * input_ids.float().mean()
        return SimpleNamespace(loss=loss)


def test_training_reference_averages_distinct_real_profile_rank_batches(reference, monkeypatch):
    """Check different rank inputs and float32 gradient averaging rather than replicated tokens."""
    monkeypatch.setattr(reference, 'fixed_documents', lambda checkpoint: ([[1, 2, 4, 8]], 16))
    model = TrainingReferenceFixture()
    result = reference.training_reference(model, 'unused', batch_size=2, seq_len=32, world_size=3)
    expected = torch.tensor(result['tokens']).float().mean().item()
    assert model.checkpointing
    assert len(result['tokens']) == 6 and all(len(row) == 32 for row in result['tokens'])
    assert len(model.batches) == 3 and not torch.equal(model.batches[0], model.batches[1])
    assert result['world_size'] == 3 and result['batch_size'] == 2 and result['seq_len'] == 32
    assert result['loss'] == pytest.approx(sum(result['loss_per_rank']) / 3)
    for probe in result['gradients'].values():
        assert probe['values'] == pytest.approx([expected] * len(probe['indices']))
    assert all(parameter.grad is None for parameter in model.parameters.values())


def test_targeted_reference_matches_full_backward_without_unneeded_gradients(reference, monkeypatch):
    """Preserve full-model derivatives while avoiding storage for unsampled parameter gradients."""
    documents = [[1, 2, 4, 8]]
    monkeypatch.setattr(reference, 'fixed_documents', lambda checkpoint: (documents, 16))
    model = TrainingReferenceFixture()
    ignored = torch.nn.Parameter(torch.tensor([2.0]))
    model.parameters['model.layers.2.unprobed.weight'] = ignored
    tokens = reference.packed_tokens(documents, 16, 32, 6)
    expected = {}
    for rank in range(2):
        model.zero_grad(set_to_none=True)
        inputs = torch.tensor(tokens[rank * 3:(rank + 1) * 3])
        model(input_ids=inputs, labels=inputs, use_cache=False).loss.backward()
        for name, parameter in model.parameters.items():
            expected[name] = expected.get(name, torch.zeros_like(parameter)) + parameter.grad.detach() / 2
    model.zero_grad(set_to_none=True)
    grad = Mock(wraps=torch.autograd.grad)
    monkeypatch.setattr(torch.autograd, 'grad', grad)
    result = reference.training_reference(model, 'unused', batch_size=3, seq_len=32, world_size=2)
    assert grad.call_count == 2
    for call in grad.call_args_list:
        assert all(parameter is not ignored for parameter in call[0][1])
    for name, probe in result['gradients'].items():
        assert probe['values'] == pytest.approx(expected[name].flatten()[probe['indices']].tolist())
    assert all(parameter.requires_grad and parameter.grad is None for parameter in model.parameters.values())


@pytest.mark.parametrize('batch_size,seq_len,world_size', [(0, 32, 1), (1, 16, 1), (1, 32, 0)])
def test_training_reference_rejects_invalid_profile(reference, batch_size, seq_len, world_size):
    """Invalid global-batch definitions must fail before any model work."""
    with pytest.raises(ValueError, match='reference profile'):
        reference.training_reference(TrainingReferenceFixture(), 'unused', batch_size, seq_len, world_size)


@pytest.fixture
def loading_contract(reference, tmp_path, monkeypatch):
    """Mock library boundaries while exercising all loader identity checks."""
    if sys.version_info < (3, 10):
        pytest.skip('The standalone model runtime requires Python 3.10 or later.')
    import importlib.metadata
    from superbench.benchmarks.model_benchmarks.deepseek_v2_lite.utils import EXPECTED_CONFIG

    (tmp_path / 'config.json').write_text(json.dumps(dict(EXPECTED_CONFIG, model_type='deepseek_v2')))
    parameter = SimpleNamespace(numel=lambda: reference.PARAMETER_COUNT, requires_grad=True)
    model = SimpleNamespace(parameters=lambda: [parameter])
    loading = {key: [] for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs']}
    loader = Mock(return_value=(model, loading))
    tokenizer = SimpleNamespace(encode=Mock(return_value=[1, 2]), eos_token_id=9)
    tokenizer_loader = Mock(return_value=tokenizer)
    monkeypatch.setitem(
        sys.modules, 'transformers',
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        )
    )
    version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, 'version', lambda name: '4.46.3' if name == 'transformers' else version(name)
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    return SimpleNamespace(
        path=tmp_path,
        model=model,
        parameter=parameter,
        loading=loading,
        loader=loader,
        tokenizer=tokenizer,
        tokenizer_loader=tokenizer_loader,
    )


def test_official_loader_contract(reference, loading_contract):
    """Require local official-code loading and complete loading diagnostics."""
    item = loading_contract
    assert reference.load_model(str(item.path), torch.bfloat16) is item.model
    options = item.loader.call_args[1]
    assert options['torch_dtype'] == torch.bfloat16
    assert options['local_files_only'] and options['trust_remote_code'] and options['use_safetensors']
    assert options['attn_implementation'] == 'eager' and options['output_loading_info']
    documents, eos = reference.fixed_documents(str(item.path))
    assert len(documents) == len(reference.FIXED_TEXTS) and eos == 9
    assert item.tokenizer_loader.call_args[1]['local_files_only']
    assert item.tokenizer.encode.call_count == len(reference.FIXED_TEXTS)


@pytest.mark.parametrize('field', ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
def test_loader_diagnostics_cannot_be_ignored(reference, loading_contract, field):
    """Partial or incompatible loading must fail before reference execution."""
    loading_contract.loading[field] = ['invalid checkpoint']
    with pytest.raises(RuntimeError, match='checkpoint loading'):
        reference.load_model(str(loading_contract.path), torch.bfloat16)


@pytest.mark.parametrize('failure', ['count', 'frozen', 'gpu', 'version'])
def test_loader_rejects_wrong_model_or_runtime(reference, loading_contract, monkeypatch, failure):
    """Do not replace full-model GPU verification with a smaller or unsupported runtime."""
    item = loading_contract
    if failure == 'count':
        item.parameter.numel = lambda: 1
    elif failure == 'frozen':
        item.parameter.requires_grad = False
    elif failure == 'gpu':
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    else:
        import importlib.metadata
        monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.0')
    with pytest.raises(RuntimeError):
        reference.load_model(str(item.path), torch.bfloat16)


@pytest.mark.parametrize('mode', ['training', 'inference'])
def test_reference_entrypoint_uses_expected_dtype_and_teardown(reference, tmp_path, monkeypatch, mode):
    """Exercise orchestration with fake model work, not GPU numerical acceptance."""
    model = Mock()
    model.cuda.return_value = model
    loader = Mock(return_value=model)
    monkeypatch.setattr(reference, 'load_model', loader)
    monkeypatch.setattr(reference, 'training_reference', Mock(return_value={'loss': 1.0}))
    monkeypatch.setattr(reference, 'inference_reference', Mock(return_value={'comparisons': []}))
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'synchronize', Mock())
    cleanup = Mock()
    monkeypatch.setattr(torch.cuda, 'empty_cache', cleanup)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '0')
    requests = tmp_path / 'requests.json'
    requests.write_text('[]')
    output = tmp_path / 'report.json'
    monkeypatch.setattr(
        sys, 'argv', [
            'reference',
            '--checkpoint',
            str(tmp_path),
            '--mode',
            mode,
            '--requests',
            str(requests),
            '--output',
            str(output),
        ]
    )
    reference.main()
    report = json.loads(output.read_text())
    expected = torch.bfloat16 if mode == 'training' else torch.float32
    assert loader.call_args[0][1] == expected
    assert report['reference_dtype'] == str(expected)
    assert report['correctness_passed'] and report['teardown_complete'] and not report['tf32_allowed']
    cleanup.assert_called_once()


def test_worker_environment_reaches_the_worker(tmp_path, monkeypatch):
    """Put this package first on the worker search path and pass it through the supervisor."""
    monkeypatch.setenv('PYTHONPATH', 'existing')
    expected = os.pathsep.join([str(Path(utils.__file__).resolve().parents[1]), 'existing'])
    command = [sys.executable, '-c', 'import os, sys; assert os.environ["PYTHONPATH"] == sys.argv[1]', expected]
    utils.run_worker(command, tmp_path / 'worker.log', 5, env=utils.worker_environment())


def test_worker_modules_import_without_superbench(tmp_path):
    """Worker interpreters lack SuperBench and site packages, so loading a worker module must need neither."""
    package = Path(utils.__file__).resolve().parent
    # The cache helper uses the SuperBench logger and runs only in the adapter process.
    names = sorted(path.stem for path in package.glob('*.py') if path.stem not in ('__init__', 'cache'))
    assert {'reference', 'utils'} <= set(names)
    code = (
        'import importlib, sys; [importlib.import_module(name) for name in sys.argv[1:]]; '
        'assert "superbench" not in sys.modules'
    )
    modules = ['{}.{}'.format(package.name, name) for name in names]
    utils.subprocess.run(
        [sys.executable, '-S', '-c', code, *modules], env=utils.worker_environment(), cwd=str(tmp_path), check=True
    )
