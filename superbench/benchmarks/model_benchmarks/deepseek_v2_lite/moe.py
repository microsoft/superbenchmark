# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stable expert dispatch for the pinned full-parameter V2-Lite training model."""

import sys
from types import MethodType

MOE_DISPATCHES = ('official', 'stable_sorted')


def sorted_training_forward(module, hidden_states, auxiliary_apply):
    """Preserve route order and every expert while avoiding repeated Boolean-index synchronizations."""
    import torch

    original_shape = hidden_states.shape
    indices, weights, auxiliary = module.gate(hidden_states)
    routed = hidden_states.reshape(-1, original_shape[-1]).repeat_interleave(module.num_experts_per_tok, dim=0)
    expert_ids = indices.reshape(-1)
    order = torch.argsort(expert_ids, stable=True)
    counts = torch.bincount(expert_ids, minlength=len(module.experts)).tolist()
    segments = torch.split(routed.index_select(0, order), counts, dim=0)
    outputs = torch.cat([expert(segment) for expert, segment in zip(module.experts, segments)], dim=0)
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device))
    routed_output = outputs.index_select(0, inverse)
    combined = (routed_output.view(*weights.shape, -1) * weights.unsqueeze(-1)).sum(dim=1)
    combined = combined.to(hidden_states.dtype).view(original_shape)
    combined = auxiliary_apply(combined, auxiliary)
    if module.config.n_shared_experts is not None:
        combined = combined + module.shared_experts(hidden_states)
    return combined


def install_sorted_dispatch(model):
    """Replace only the 26 pinned MoE training forwards, leaving evaluation and parameters intact."""
    modules = [module for module in model.modules() if type(module).__name__ == 'DeepseekV2MoE']
    if len(modules) != 26:
        raise ValueError('Expected all 26 official MoE layers, found {}.'.format(len(modules)))
    if any(module.ep_size != 1 for module in modules):
        raise ValueError('Stable expert dispatch does not implement expert parallelism.')
    for module in modules:
        original = module.forward
        auxiliary_apply = sys.modules[type(module).__module__].AddAuxiliaryLoss.apply

        def forward(self, hidden_states, original=original, auxiliary_apply=auxiliary_apply):
            if not self.training:
                return original(hidden_states)
            return sorted_training_forward(self, hidden_states, auxiliary_apply)

        module.forward = MethodType(forward, module)
    return len(modules)
