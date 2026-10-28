# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU mathematical and gradient controls for stable expert dispatch."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if sys.version_info < (3, 10):
    pytest.skip('The standalone V2-Lite training runtime requires Python 3.10 or later.', allow_module_level=True)
if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip('torch')
SPEC = importlib.util.spec_from_file_location(
    'sorted_dispatch',
    Path(__file__).resolve().parents[3] / 'superbench/benchmarks/model_benchmarks/deepseek_v2_lite_moe.py'
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError('Cannot load the stable dispatch implementation.')
dispatch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatch)


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
    actual = dispatch.sorted_training_forward(model, values, Auxiliary.apply)
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected_output, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(values.grad, expected_input, rtol=1e-12, atol=1e-12)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected_parameters[name], rtol=1e-12, atol=1e-12)
    if empty_expert:
        assert torch.count_nonzero(model.experts[-1][0].weight.grad).item() == 0


def test_installation_preserves_parameters_and_the_evaluation_path(monkeypatch):
    """Only training dispatch changes; evaluation retains the original forward."""
    model = torch.nn.ModuleList([DeepseekV2MoE(False, True).double() for _ in range(26)])
    parameters = [id(parameter) for parameter in model.parameters()]
    values = torch.randn(1, 2, 4, dtype=torch.float64)
    model.eval()
    expected = model[0](values).detach()
    assert dispatch.install_sorted_dispatch(model) == 26
    assert [id(parameter) for parameter in model.parameters()] == parameters
    replacement = Mock(side_effect=RuntimeError('training dispatch reached'))
    monkeypatch.setattr(dispatch, 'sorted_training_forward', replacement)
    torch.testing.assert_close(model[0](values), expected)
    replacement.assert_not_called()
    model.train()
    with pytest.raises(RuntimeError, match='training dispatch reached'):
        model[0](values)


@pytest.mark.parametrize('count,expert_parallelism', [(25, 1), (27, 1), (26, 2)])
def test_unsupported_layout_is_rejected_before_any_forward_changes(count, expert_parallelism):
    """A partial model or expert-parallel layout cannot receive a partial patch."""
    model = torch.nn.ModuleList([DeepseekV2MoE(False, False) for _ in range(count)])
    model[0].ep_size = expert_parallelism
    original = [module.forward for module in model]
    with pytest.raises(ValueError):
        dispatch.install_sorted_dispatch(model)
    assert [module.forward for module in model] == original
