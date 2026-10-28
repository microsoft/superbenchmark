# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU controls for the input-matched policy, not hardware qualification."""

import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def numerics(monkeypatch):
    """Load the standalone reference without importing the engine."""
    directory = Path(__file__).resolve().parents[3] / 'superbench/benchmarks/model_benchmarks'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('v2lite_numerics_test', directory / 'deepseek_v2_lite_numerics.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the standalone numerical reference.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def budgets():
    """Use deliberately small synthetic bounds for arithmetic controls."""
    return {
        'max_probability_error': 0.02,
        'relative_probability_mass_error': 0.03,
        'normalized_logprob_error': 0.5,
    }


@pytest.fixture
def policy_file(numerics, tmp_path, monkeypatch):
    """Create qualified-shaped test evidence without claiming GPU execution."""
    monkeypatch.setattr(numerics, '__file__', str(tmp_path / 'superbench/benchmarks/model_benchmarks/reference.py'))
    path = tmp_path / 'superbench/config/deepseek-v2-lite-h100-numerics.json'
    path.parent.mkdir(parents=True)
    candidate = {
        'method': numerics.METHOD,
        'reference_only': True,
        'input_matched_limit_rule': numerics.LIMIT_RULE,
        'qualification_offsets': list(numerics.OFFSETS),
        'profile': {key: numerics.PROFILE[key]
                    for key in ('batch_size', 'seq_len', 'max_new_tokens')},
        'qualification_stream': {key: numerics.PROFILE[key]
                                 for key in ('num_warmup', 'num_steps')},
    }
    candidate['frozen_limits_sha256'] = numerics.canonical_hash(candidate)
    contract = {
        'status': 'qualified',
        'calibrated_contract': candidate,
        'qualification': [{
            'profile_offset': offset
        } for offset in numerics.OFFSETS],
        'reference_validation': {
            'non_anchor_comparisons': 1
        },
    }

    def save():
        path.write_text(json.dumps(contract))
        monkeypatch.setattr(numerics, 'CONTRACT_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())

    save()
    return SimpleNamespace(path=path, contract=contract, candidate=candidate, save=save)


def test_frozen_profile_is_explicit_and_legacy_policy_is_unchanged(numerics):
    """Do not apply one H100 qualification to other shapes or the default policy."""
    numerics.validate_profile(SimpleNamespace(numerical_policy='strict_fp32', batch_size=1))
    args = SimpleNamespace(numerical_policy=numerics.POLICY, **numerics.PROFILE)
    numerics.validate_profile(args)
    args.num_steps -= 1
    with pytest.raises(ValueError, match='complete frozen'):
        numerics.validate_profile(args)
    with pytest.raises(ValueError, match='Unknown'):
        numerics.validate_profile(SimpleNamespace(numerical_policy='other'))


def test_contract_requires_installed_qualification(numerics, monkeypatch):
    """An unfinished policy cannot start a reference workload."""
    monkeypatch.setattr(numerics, 'CONTRACT_SHA256', '')
    with pytest.raises(RuntimeError, match='qualified contract'):
        numerics.load_contract()


def test_contract_is_bound_to_exact_file_bytes(numerics, policy_file):
    """Even a semantically equivalent substituted policy requires a new identity."""
    assert numerics.load_contract() == policy_file.contract
    policy_file.path.write_text(policy_file.path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        numerics.load_contract()


@pytest.mark.parametrize('fault', ['status', 'method', 'shape', 'stream', 'offsets', 'evidence', 'anchor-only', 'hash'])
def test_contract_rejects_incomplete_qualification(numerics, policy_file, fault):
    """Validate semantics as well as a matching artifact digest."""
    candidate, contract = policy_file.candidate, policy_file.contract
    if fault == 'status':
        contract['status'] = 'candidate'
    elif fault == 'method':
        candidate['method'] = 'unqualified-method'
    elif fault == 'shape':
        candidate['profile']['batch_size'] = 1
    elif fault == 'stream':
        candidate['qualification_stream']['num_steps'] = 1
    elif fault == 'offsets':
        candidate['qualification_offsets'].pop()
    elif fault == 'evidence':
        contract['qualification'].pop()
    elif fault == 'anchor-only':
        contract['reference_validation']['non_anchor_comparisons'] = 0
    else:
        candidate['frozen_limits_sha256'] = 'changed'
    if fault != 'hash':
        candidate['frozen_limits_sha256'] = numerics.canonical_hash(
            {key: value
             for key, value in candidate.items() if key != 'frozen_limits_sha256'}
        )
    policy_file.save()
    with pytest.raises(ValueError, match='qualification'):
        numerics.load_contract()


@pytest.mark.parametrize(
    'expected,actual', [
        ([], []),
        ([-1.0], []),
        ([float('nan')], [-1.0]),
        ([-1.0], [float('inf')]),
        ([-1.0], [True]),
        ([-1.0], [0.1]),
        ([-1000.0], [-1000.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ]
)
def test_invalid_probability_data_is_rejected(numerics, expected, actual):
    """Reject malformed, nonfinite and unnormalized probabilities explicitly."""
    with pytest.raises(ValueError):
        numerics.probability_errors(expected, actual)


def test_probability_mass_and_log_metrics_retain_every_token(numerics):
    """Detect normalization defects even when the greedy token is unchanged."""
    expected = [math.log(0.6), math.log(0.3)]
    actual = [value - math.log(2) for value in expected]
    errors = numerics.probability_errors(expected, actual)
    assert errors['max_probability_error'] == pytest.approx(0.3)
    assert errors['relative_probability_mass_error'] == pytest.approx(0.5)
    assert errors['max_logprob_error'] == pytest.approx(math.log(2))
    assert errors['normalized_logprob_error'] > 0


def test_input_matched_limits_are_reference_only(numerics):
    """Keep calibration unchanged and fail if uncertainty admits half normalization."""
    calibrated = budgets()
    original = dict(calibrated)
    reference = {metric: 0.01 for metric in numerics.METRICS}
    assert numerics.input_matched_limits(calibrated, reference) == {
        metric: original[metric] + reference[metric]
        for metric in numerics.METRICS
    }
    assert calibrated == original
    reference['relative_probability_mass_error'] = 0.5
    with pytest.raises(ValueError, match='half-normalization'):
        numerics.input_matched_limits(calibrated, reference)
    reference['relative_probability_mass_error'] = float('nan')
    with pytest.raises(ValueError, match='input-matched'):
        numerics.input_matched_limits(calibrated, reference)
    reference['relative_probability_mass_error'] = 0
    calibrated['normalized_logprob_error'] = 0
    with pytest.raises(ValueError, match='budget'):
        numerics.input_matched_limits(calibrated, reference)


def test_greedy_rules_and_invalid_bounds(numerics):
    """Near ties do not waive reported-greedy or probability-gap checks."""
    expected = {0: math.log(0.9), 1: math.log(0.1)}
    result = numerics.score_probabilities(expected, expected, 1, 0, budgets())
    assert not result['accepted'] and not result['greedy_match']
    assert 'reported token is not greedy' in result['failures']
    assert 'greedy choice exceeds calibrated uncertainty' in result['failures']
    assert numerics.score_probabilities(expected, expected, 0, 0, budgets())['accepted']
    with pytest.raises(ValueError, match='Incomplete'):
        numerics.score_probabilities(expected, {0: expected[0]}, 0, 0, budgets())
    limits = budgets()
    limits['normalized_logprob_error'] = float('inf')
    with pytest.raises(ValueError, match='bound'):
        numerics.score_probabilities(expected, expected, 0, 0, limits)


def test_top_k_shape_bounds_and_tail_fault(numerics):
    """A tiny tail still has to reject the fixed temperature fault."""
    torch = pytest.importorskip('torch')
    logits = torch.full((32, ), -40.0)
    logits[0] = 0
    expected = logits.log_softmax(-1)
    wrong = (expected / 2).log_softmax(-1)
    ids = wrong.topk(20).indices.tolist()
    observed = {str(key): wrong[key].item() for key in ids}
    result = numerics.evaluate_logits(expected, observed, 0, budgets())
    assert not result['accepted'] and 'normalized_logprob_error' in result['failures']
    with pytest.raises(ValueError, match='top-20'):
        numerics.evaluate_logits(expected, {'0': 0.0}, 0, budgets())
    observed['1000'] = observed.pop(str(ids[-1]))
    with pytest.raises(ValueError, match='out of range'):
        numerics.evaluate_logits(expected, observed, 0, budgets())
    missing = {str(key): -10.0 for key in range(1, 21)}
    assert not numerics.evaluate_logits(expected, missing, 1, budgets())['accepted']
    duplicate = {str(key): -10.0 for key in range(20)}
    duplicate['00'] = -10.0
    with pytest.raises(ValueError, match='duplicated'):
        numerics.evaluate_logits(expected, duplicate, 0, budgets())


@pytest.fixture
def cpu_reference(monkeypatch):
    """Exercise reference tensor operations on CPU with explicit fake device transport."""
    torch = pytest.importorskip('torch')
    tensor, ones = torch.tensor, torch.ones

    def cpu_tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(*args, **kwargs)

    def cpu_ones(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return ones(*args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    monkeypatch.setattr(torch, 'ones', cpu_ones)
    return torch


def test_complete_prefix_and_cached_reference_shapes(numerics, cpu_reference):
    """Check all attention-mask lengths and cache positions without a GPU."""
    torch = cpu_reference
    calls = []

    def model(input_ids, attention_mask, use_cache, past_key_values=None):
        calls.append((input_ids.shape, attention_mask.shape, use_cache, past_key_values))
        logits = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24)
        return SimpleNamespace(
            logits=logits.expand(input_ids.shape[0], input_ids.shape[1], 24), past_key_values=len(calls)
        )

    cases = [{'id': str(index), 'prompt_token_ids': [1, 2, 3], 'continuation_ids': [4, 5]} for index in range(3)]
    full = numerics.full_logits(model, [[1, 2, 3]])
    assert full.shape == (1, 24) and calls[-1][2:] == (False, None)
    values = numerics.collect_matched_bf16(model, cases, 2)
    assert set(values) == {'0', '1', '2'} and all(value.shape == (2, 24) for value in values.values())
    assert calls[1][0:2] == (torch.Size([2, 3]), torch.Size([2, 3]))
    assert calls[2][0:2] == (torch.Size([2, 1]), torch.Size([2, 4]))
    assert calls[2][3] is not None
    errors = numerics.tensor_errors(values['0'][0], values['1'][0])
    assert errors['greedy_match'] and all(errors[metric] == 0 for metric in numerics.METRICS)
    with pytest.raises(ValueError, match='uniquely'):
        numerics.configured_batches(cases + cases, 2)
    mismatched = [cases[0], dict(cases[1], continuation_ids=[4])]
    with pytest.raises(ValueError, match='matching sequence'):
        list(numerics.cached_logits(model, mismatched))


def test_reference_rejects_nonfinite_and_incomplete_output(numerics, cpu_reference, monkeypatch):
    """Reference failures cannot create success-shaped numerical evidence."""
    torch = cpu_reference

    def model(**kwargs):
        return SimpleNamespace(logits=torch.full((1, 2, 24), float('nan')), past_key_values=1)

    cases = [{'id': '0', 'prompt_token_ids': [1, 2], 'continuation_ids': [3]}]
    with pytest.raises(RuntimeError, match='Nonfinite'):
        numerics.full_logits(model, [[1, 2]])
    with pytest.raises(RuntimeError, match='Nonfinite'):
        list(numerics.cached_logits(model, cases))
    monkeypatch.setattr(numerics, 'cached_logits', lambda *args: iter([]))
    with pytest.raises(ValueError, match='Incomplete'):
        numerics.collect_matched_bf16(model, cases, 1)
    monkeypatch.setattr(numerics, 'cached_logits', lambda *args: iter([(1, torch.zeros((1, 24)))]))
    with pytest.raises(ValueError, match='out-of-order'):
        numerics.collect_matched_bf16(model, cases, 1)


def request_sets(numerics):
    """Create token-shaped fixtures, not reference or engine measurements."""
    documents, eos, records = [[1, 2, 3], [4, 5, 6]], 99, []
    short = [
        numerics.packed_tokens(documents, eos, length, 1, index * 7)[0] for index, length in enumerate((16, 23, 32))
    ]
    for offset in numerics.OFFSETS:
        prompts = short * 2 + numerics.packed_tokens(documents, eos, 512, 8, offset)
        for index, prompt in enumerate(prompts):
            count = 4 if index < 6 else 64
            row = {
                'prompt_token_ids': prompt,
                'token_ids': [1] * count,
                'logprobs': [{} for _ in range(count)],
                'numerical_set_offset': offset,
            }
            if index >= 6:
                row.update(profile_request_index=index - 6, profile_batch_size=8)
            records.append(row)
    return documents, eos, records


@pytest.mark.parametrize('fault', ['missing-window', 'wrong-prompt', 'short-decode', 'lost-row', 'bad-token', 'trace'])
def test_frozen_request_sets_cannot_drop_positions(numerics, fault):
    """Require all six windows and every configured-batch row."""
    documents, eos, requests = request_sets(numerics)
    groups = numerics.validate_requests(requests, documents, eos)
    assert list(groups) == list(numerics.OFFSETS)
    if fault == 'missing-window':
        requests.pop()
    elif fault == 'wrong-prompt':
        requests[0]['prompt_token_ids'] = [7]
    elif fault == 'short-decode':
        for row in requests[:14]:
            row['token_ids'] = row['token_ids'][:1]
            row['logprobs'] = row['logprobs'][:1]
    elif fault == 'lost-row':
        requests[6]['profile_request_index'] = 2
    elif fault == 'bad-token':
        requests[0]['token_ids'][0] = True
    else:
        requests[0]['logprobs'].pop()
    with pytest.raises(ValueError):
        numerics.validate_requests(requests, documents, eos)


@pytest.mark.parametrize('corrupt', [False, True])
def test_group_scoring_retains_all_positive_and_fault_positions(numerics, monkeypatch, corrupt):
    """A good synthetic pair must reject every fault; a corrupted pair must fail."""
    torch = pytest.importorskip('torch')
    vectors = [torch.arange(32, dtype=torch.float32), torch.arange(31, -1, -1, dtype=torch.float32)]
    expected = [value.log_softmax(-1) for value in vectors]
    requests, matched = [], {}
    for index in range(2):
        ids = expected[index].topk(20).indices.tolist()
        observed = {str(key): expected[index][key].item() for key in ids}
        if corrupt:
            observed = {key: value - math.log(2) for key, value in observed.items()}
        requests.append(
            {
                'prompt_token_ids': [index],
                'token_ids': [expected[index].argmax().item()] * 2,
                'logprobs': [observed, observed],
            }
        )
        matched['request-{}'.format(index)] = torch.stack([expected[index], expected[index]])
    monkeypatch.setattr(numerics, 'full_logits', lambda model, rows: vectors[rows[0][0]].unsqueeze(0))
    result = numerics.evaluate_group(None, requests, matched, budgets())
    assert result['positive_positions'] == 4 and result['negative_positions'] == 9
    assert len(result['positive_failures']) == (4 if corrupt else 0)
    assert not result['negative_failures']
    assert {row['fault']
            for row in result['all_negative_records']} == {
                'half-normalization',
                'temperature-two',
                'last-request-context-for-first',
            }


@pytest.mark.parametrize('failure', [False, True])
def test_reference_models_are_sequential_and_always_released(numerics, monkeypatch, failure):
    """Exercise lifecycle plumbing with explicit CPU-only model doubles."""
    if sys.version_info < (3, 10):
        pytest.skip('The standalone reference runtime requires Python 3.10 or later.')
    torch = pytest.importorskip('torch')
    import deepseek_v2_lite_reference as reference

    loaded = []

    def load_model(checkpoint, dtype):
        loaded.append(dtype)
        model = SimpleNamespace()
        model.cuda = lambda: model
        model.eval = lambda: model
        return model

    monkeypatch.setattr(reference, 'load_model', load_model)
    monkeypatch.setattr(reference, 'fixed_documents', lambda checkpoint: ([[1, 2]], 99))
    monkeypatch.setattr(numerics, 'validate_requests', lambda *args: {0: [{'prompt_token_ids': [1], 'token_ids': [2]}]})
    monkeypatch.setattr(numerics, 'collect_matched_bf16', lambda *args: {})
    monkeypatch.setattr(torch, 'set_num_threads', Mock())
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', Mock())
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'synthetic-h100')
    release = Mock()
    monkeypatch.setattr(torch.cuda, 'empty_cache', release)
    monkeypatch.setattr(torch.cuda, 'synchronize', Mock())
    from importlib.metadata import version
    identity = {
        'torch_version': torch.__version__,
        'transformers_version': version('transformers'),
        'gpu_name': 'synthetic-h100',
        'reference_loader_sha256': hashlib.sha256(inspect.getsource(load_model).encode()).hexdigest(),
        'tf32_allowed': False,
        'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
    contract = {
        'calibrated_contract': {
            'runtime_identity': identity,
            'limits': budgets(),
            'frozen_limits_sha256': 'fixture'
        }
    }
    evaluation = Mock(
        side_effect=RuntimeError('reference failed') if failure else None,
        return_value={
            'positive_positions': 1,
            'negative_positions': 3,
            'positive_failures': [],
            'negative_failures': []
        },
    )
    monkeypatch.setattr(numerics, 'evaluate_group', evaluation)
    if failure:
        with pytest.raises(RuntimeError, match='reference failed'):
            numerics.run_reference(Path('fixture'), [1], contract)
    else:
        result = numerics.run_reference(Path('fixture'), [1], contract)
        assert result['correctness_passed'] is True and result['teardown_complete'] is True
    assert loaded == [torch.bfloat16, torch.float32]
    assert release.call_count == 2
    identity['gpu_name'] = 'different-hardware'
    with pytest.raises(ValueError, match='runtime or hardware'):
        numerics.run_reference(Path('fixture'), [1], contract)


@pytest.mark.parametrize('passed', [True, False])
def test_reference_cli_preserves_rejection_evidence(numerics, monkeypatch, tmp_path, passed):
    """A rejected report is saved but the process must still fail explicitly."""
    requests, output = tmp_path / 'requests.json', tmp_path / 'reference.json'
    requests.write_text('[]')
    monkeypatch.setattr(
        sys, 'argv', [
            'reference',
            '--checkpoint',
            str(tmp_path),
            '--requests',
            str(requests),
            '--output',
            str(output),
        ]
    )
    monkeypatch.setattr(numerics, 'load_contract', lambda: {'fixture': True})
    monkeypatch.setattr(numerics, 'run_reference', lambda *args: {'correctness_passed': passed})
    validate = Mock()
    monkeypatch.setattr(numerics, 'validate_reference_result', validate)
    if passed:
        numerics.main()
    else:
        with pytest.raises(RuntimeError, match='rejected'):
            numerics.main()
    assert json.loads(output.read_text())['correctness_passed'] is passed
    assert numerics.os.environ['NVIDIA_TF32_OVERRIDE'] == '0'
    assert validate.call_count == int(passed)


def serialized_reference(numerics):
    """Build complete control-flow evidence with explicit synthetic accepted/rejected flags."""
    windows = {}
    for offset in numerics.OFFSETS:
        positives, negatives = [], []
        for request in range(14):
            count = 4 if request < 6 else 64
            for step in range(count):
                positives.append({'request': request, 'step': step, 'accepted': True})
                negatives.extend(
                    [
                        {
                            'request': request,
                            'step': step,
                            'accepted': False,
                            'fault': fault
                        } for fault in ('half-normalization', 'temperature-two')
                    ]
                )
        negatives.append({'request': 0, 'step': 0, 'accepted': False, 'fault': 'last-request-context-for-first'})
        windows[str(offset)] = {
            'positive_positions': len(positives),
            'negative_positions': len(negatives),
            'positive_failures': [],
            'negative_failures': [],
            'all_positive_records': positives,
            'all_negative_records': negatives,
        }
    return {
        'correctness_passed': True,
        'teardown_complete': True,
        'numerical_policy': numerics.POLICY,
        'contract_sha256': numerics.CONTRACT_SHA256,
        'profile_offsets': list(numerics.OFFSETS),
        'request_count': 84,
        'positive_positions': 3216,
        'negative_positions': 6438,
        'windows': windows,
    }


@pytest.mark.parametrize(
    'fault', [
        'identity',
        'missing-window',
        'invalid-window',
        'missing-position',
        'accepted-fault',
        'failed-positive',
        'bad-index',
        'duplicate-position',
        'missing-step',
        'wrong-fault',
        'totals',
    ]
)
def test_serialized_reference_cannot_fabricate_complete_success(numerics, fault):
    """Validate all per-window records rather than trusting a summary flag."""
    report = serialized_reference(numerics)
    numerics.validate_reference_result(report)
    window = report['windows']['0']
    overrides = {'identity': ('contract_sha256', 'another-policy'), 'totals': ('positive_positions', 3215)}
    if fault in overrides:
        key, value = overrides[fault]
        report[key] = value
    elif fault == 'missing-window':
        report['windows'].pop('0')
    elif fault == 'invalid-window':
        report['windows']['0'] = None
    elif fault == 'missing-position':
        window['all_positive_records'].pop()
    elif fault == 'accepted-fault':
        window['all_negative_records'][0]['accepted'] = True
    elif fault == 'failed-positive':
        window['all_positive_records'][0]['accepted'] = False
    elif fault == 'bad-index':
        window['all_positive_records'][0]['step'] = True
    elif fault == 'duplicate-position':
        window['all_positive_records'][0] = dict(window['all_positive_records'][1])
    elif fault == 'missing-step':
        window['all_positive_records'][0]['step'] = 5
    else:
        window['all_negative_records'][-1]['fault'] = 'substituted-fault'
    with pytest.raises(ValueError):
        numerics.validate_reference_result(report)
