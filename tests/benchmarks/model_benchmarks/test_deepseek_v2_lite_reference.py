# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU control-flow checks, not certification of full-model reference numerics."""

import importlib.util
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip('torch')


@pytest.fixture
def reference(monkeypatch):
    """Load the standalone oracle and redirect only test tensor creation to CPU."""
    directory = Path(__file__).resolve().parents[3] / 'superbench/benchmarks/model_benchmarks'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('v2lite_reference_test', directory / 'deepseek_v2_lite_reference.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the standalone reference.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tensor = torch.tensor

    def cpu_tensor(data, *args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(data, *args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    return module


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


def test_long_decode_checks_every_engine_token_with_bounded_full_prefix_calls(reference):
    """Only full-prefix recomputation is sampled, never engine probability validation."""
    request = request_fixture()
    request['token_ids'] *= 4
    request['logprobs'] *= 4
    request['full_prefix_steps'] = [0, 1, 4, 7]
    model = ReferenceFixture()
    result = reference.inference_reference(model, [request])
    assert len(result['comparisons']) == 8
    assert len(model.calls) == 12
    assert [row['step'] for row in result['comparisons'] if row['full_prefix_checked']] == [0, 1, 4, 7]
    assert all(row['cache_max_abs_error'] is None for row in result['comparisons'] if not row['full_prefix_checked'])
    request['logprobs'][2] = dict(request['logprobs'][2], **{'3': 0.0})
    with pytest.raises(AssertionError, match='request 0 step 2'):
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


def test_reference_preserves_exact_rank_microbatch_shape(reference, monkeypatch):
    """Do not introduce BF16 batch-regrouping differences while checking distributed gradients."""
    monkeypatch.setattr(reference, 'fixed_documents', lambda checkpoint: ([[1, 2, 4, 8]], 16))
    model = TrainingReferenceFixture()
    result = reference.training_reference(model, 'unused', batch_size=3, seq_len=32, world_size=2)
    assert [batch.shape[0] for batch in model.batches] == [3, 3]
    assert all(batch.shape[1] == 32 for batch in model.batches)
    assert torch.equal(torch.cat(model.batches), torch.tensor(result['tokens']))
    expected = torch.tensor(result['tokens']).float().mean().item()
    assert result['reference_microbatch_size'] == 3
    assert result['loss'] == pytest.approx(expected * 24)
    for probe in result['gradients'].values():
        assert probe['values'] == pytest.approx([expected] * len(probe['indices']))


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


def test_reference_does_not_retain_unused_model_outputs(reference, monkeypatch):
    """Release returned vocabulary tensors before accumulating full-model gradients."""
    monkeypatch.setattr(reference, 'fixed_documents', lambda checkpoint: ([[1, 2, 4, 8]], 16))

    class OutputFixture(TrainingReferenceFixture):
        def __call__(self, input_ids, labels, use_cache):
            output = super().__call__(input_ids, labels, use_cache)
            logits = torch.ones(8, 8)
            self.logits_reference = weakref.ref(logits)
            return SimpleNamespace(loss=output.loss, logits=logits)

    model = OutputFixture()

    def check_released(gradient):
        assert model.logits_reference() is None
        return gradient

    next(iter(model.parameters.values())).register_hook(check_released)
    reference.training_reference(model, 'unused', batch_size=2, seq_len=32, world_size=2)


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
    from deepseek_v2_lite_utils import EXPECTED_CONFIG

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
