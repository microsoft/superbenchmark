# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU update-verification checks; full-model capacity needs real GPU calibration."""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

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
    reference = Mock(return_value={'world_size': 2, 'batch_size': 2, 'seq_len': 64})
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
    assert 0 < report['total_seconds'] < 60


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
