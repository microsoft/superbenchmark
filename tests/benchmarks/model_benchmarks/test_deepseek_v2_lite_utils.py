# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU-only tests for the standalone worker contract, without framework imports."""

import copy
import importlib.util
import math
import sys
from pathlib import Path

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
        sys.executable, '-c',
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
        [sys.executable, '-c', 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])',
         str(output), value],
        tmp_path / 'worker.log', 5
    )
    assert output.read_text() == value


def test_success_with_live_child_is_rejected(tmp_path):
    """A leader's exit code zero is insufficient when its child remains alive."""
    command = [
        sys.executable, '-c',
        'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])',
    ]
    with pytest.raises(RuntimeError, match='subprocess group'):
        utils.run_worker(command, tmp_path / 'worker.log', 5)
