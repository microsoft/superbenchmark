# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU control-flow checks, not certification of full-model reference numerics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
