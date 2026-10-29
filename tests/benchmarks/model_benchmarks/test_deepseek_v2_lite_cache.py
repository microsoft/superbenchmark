# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for bounded removal of one V2-Lite run's model cache."""

import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from superbench.benchmarks.model_benchmarks import deepseek_v2_lite_cache as cleanup


def test_removes_only_the_run_cache(tmp_path):
    """Delete the downloaded checkpoint while keeping the run's report and logs."""
    cache = tmp_path / 'cache'
    (cache / 'checkpoint').mkdir(parents=True)
    (cache / 'checkpoint' / 'model-00001-of-000004.safetensors').write_bytes(b'weights')
    (tmp_path / 'report.json').write_text('{}')
    cleanup.remove_model_cache(cache, time.monotonic() + 30)
    assert not cache.exists()
    assert (tmp_path / 'report.json').is_file()


def test_missing_cache_is_not_an_error(tmp_path):
    """A run that failed before downloading has no cache to remove."""
    cleanup.remove_model_cache(tmp_path / 'cache', time.monotonic() + 30)
    assert not (tmp_path / 'cache').exists()


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
