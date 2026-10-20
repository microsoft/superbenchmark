# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU update-verification checks; full-model capacity needs real GPU calibration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip('torch')


@pytest.fixture
def worker(monkeypatch):
    """Load the worker and replace only GPU/distributed synchronization in CPU tests."""
    directory = Path(__file__).resolve().parents[3] / 'superbench/benchmarks/model_benchmarks'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        'v2lite_training_test', directory / 'deepseek_v2_lite_training_worker.py'
    )
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the standalone training worker.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda tensor: None)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: 1)
    tensor = torch.tensor

    def cpu_tensor(data, *args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    return module


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
