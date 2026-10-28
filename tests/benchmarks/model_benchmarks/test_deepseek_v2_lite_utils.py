# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only tests for the standalone worker contract, without framework imports."""

import copy
import ctypes
import hashlib
import importlib.util
import math
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

WORKER_PATH = Path(__file__).resolve().parents[3] / 'superbench/benchmarks/model_benchmarks/deepseek_v2_lite_utils.py'
SPEC = importlib.util.spec_from_file_location('deepseek_v2_lite_utils_test', WORKER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError('Cannot load the standalone worker utilities.')
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


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


@pytest.mark.parametrize('field,value', [('seq_aux', False), ('aux_loss_alpha', 0.1), ('attention_dropout', 0.1)])
def test_fixed_training_semantics_are_part_of_model_identity(field, value):
    """Do not silently change the official auxiliary-loss or dropout configuration."""
    config = dict(utils.EXPECTED_CONFIG, model_type='deepseek_v2')
    config[field] = value
    with pytest.raises(ValueError, match=field):
        utils.validate_config(config)


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


@pytest.mark.parametrize('values', [[math.nan], [math.inf], [-1], [0], [True], [], ['1']])
def test_invalid_metric_series(values):
    """Never accept NaNs, sentinels, incomplete samples or success-shaped defaults."""
    with pytest.raises(ValueError):
        utils.numeric_series({'step_ms': values}, 'step_ms', 1)


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


def test_worker_argv_is_not_a_shell(tmp_path):
    """Preserve argument boundaries rather than reinterpret shell metacharacters."""
    output = tmp_path / 'argv'
    value = 'a=b; this is one value'
    utils.run_worker(
        [
            sys.executable, '-c', 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])',
            str(output), value
        ], tmp_path / 'worker.log', 5
    )
    assert output.read_text() == value


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
        'from deepseek_v2_lite_utils import run_worker; '
        'run_worker([sys.executable, "-c", {!r}], Path({!r}), 60)'
    ).format(str(WORKER_PATH.parent), worker, str(tmp_path / 'worker.log'))
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


def test_checkpoint_files_are_verified(checkpoint_files):
    """Check file sizes, index membership and SHA256 without claiming model execution."""
    root, _ = checkpoint_files
    utils.verify_checkpoint(root)


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


@pytest.mark.parametrize('length,count', [(0, 2), (1, 1)])
def test_invalid_probe_dimensions_fail(length, count):
    """Reject undefined probe endpoints instead of silently selecting positions."""
    with pytest.raises(ValueError, match='tensor length'):
        utils.uniform_indices(length, count)


@pytest.mark.parametrize('timeout', [0, -1, float('nan')])
def test_invalid_worker_deadline_fails_before_launch(tmp_path, timeout):
    """A nonfinite or expired budget must not start a worker."""
    with pytest.raises(TimeoutError):
        utils.run_worker([sys.executable, '-c', 'pass'], tmp_path / 'worker.log', timeout)


def test_empty_worker_command_fails(tmp_path):
    """Do not launch a supervisor with no defined workload."""
    with pytest.raises(ValueError, match='command'):
        utils.run_worker([], tmp_path / 'worker.log', 1)


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
