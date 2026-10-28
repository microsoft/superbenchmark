# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unsharded official-code reference, executed in a separate pinned environment."""

import argparse
import gc
import os
from pathlib import Path

from deepseek_v2_lite_utils import (
    FIXED_TEXTS, PARAMETER_COUNT, packed_tokens, read_json, uniform_indices, validate_config, write_json
)

TRAINING_GRADIENT_PARAMETERS = {
    'model.embed_tokens.weight',
    'model.layers.0.self_attn.q_proj.weight',
    'model.layers.0.self_attn.kv_a_proj_with_mqa.weight',
    'model.layers.0.mlp.down_proj.weight',
    'model.layers.1.mlp.gate.weight',
}


def load_model(checkpoint, dtype):
    """Load official causal-LM weights with no quantization or model reduction."""
    from importlib.metadata import version

    import torch
    from transformers import AutoModelForCausalLM

    if version('transformers') != '4.46.3':
        raise RuntimeError('The official-code reference requires transformers==4.46.3.')
    validate_config(read_json(Path(checkpoint) / 'config.json'))
    model, loading = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation='eager',
        output_loading_info=True,
    )
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if key not in loading or loading[key]:
            raise RuntimeError('Official checkpoint loading failed: {}={!r}.'.format(key, loading.get(key)))
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError('Loaded parameter count differs from the complete official V2-Lite.')
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError('The full-parameter profile must not freeze model parameters.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm GPU unavailable in the reference environment.')
    return model


def fixed_documents(checkpoint):
    """Tokenize the small, versioned natural-text input with the official tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
    return [tokenizer.encode(text, add_special_tokens=False) for text in FIXED_TEXTS], tokenizer.eos_token_id


def _training_gradient_probes(model, gradients):
    """Select fixed dense/MLA/router/active-expert positions from the first rank batch."""
    import torch

    required = TRAINING_GRADIENT_PARAMETERS
    probes = {}
    expert_probe = False
    for name, parameter in model.named_parameters():
        gradient = gradients.get(name)
        if gradient is None:
            continue
        selected_expert = (
            not expert_probe and name.startswith('model.layers.1.mlp.experts.') and name.endswith('down_proj.weight')
            and torch.count_nonzero(gradient).item() > 0
        )
        if name in required or selected_expert:
            flat = gradient.detach().flatten()
            nonzero = torch.nonzero(flat, as_tuple=False).flatten()[:512]
            spread = torch.tensor(uniform_indices(flat.numel()), device=flat.device, dtype=torch.long)
            indices = torch.unique(torch.cat((nonzero, spread)))
            values = flat[indices].float().cpu()
            if not torch.isfinite(values).all().item():
                raise RuntimeError('Nonfinite reference gradient: {}'.format(name))
            probes[name] = {
                'shape': list(parameter.shape),
                'indices': indices.cpu().tolist(),
                'values': torch.zeros_like(values)
            }
            if selected_expert:
                expert_probe = True
    if not required.issubset(probes) or not expert_probe:
        raise RuntimeError('Required dense/MLA/router/expert reference gradients are missing.')
    return probes


def training_reference(model, checkpoint, batch_size=1, seq_len=32, world_size=1):
    """Reference exact rank batches without storing unrequested parameter gradients."""
    import torch

    if batch_size < 1 or seq_len < 32 or world_size < 1:
        raise ValueError('Invalid training reference profile.')
    documents, eos = fixed_documents(checkpoint)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tokens = packed_tokens(documents, eos, seq_len, batch_size * world_size)
    losses, probes = [], {}
    targets = {
        name: parameter
        for name, parameter in model.named_parameters() if name in TRAINING_GRADIENT_PARAMETERS or
        (name.startswith('model.layers.1.mlp.experts.') and name.endswith('down_proj.weight'))
    }
    for rank in range(world_size):
        batch = tokens[rank * batch_size:(rank + 1) * batch_size]
        inputs = torch.tensor(batch, device='cuda', dtype=torch.long)
        model.zero_grad(set_to_none=True)
        output = model(input_ids=inputs, labels=inputs, use_cache=False)
        loss = output.loss
        del output
        if loss is None or not torch.isfinite(loss).item():
            raise RuntimeError('Nonfinite or missing reference causal-LM loss.')
        losses.append(loss.item())
        gradients = dict(zip(targets, torch.autograd.grad(loss, tuple(targets.values()), allow_unused=True)))
        if not probes:
            probes = _training_gradient_probes(model, gradients)
        for name, probe in probes.items():
            gradient = gradients[name]
            if gradient is None:
                if '.experts.' not in name:
                    raise RuntimeError('Missing reference gradient: {}'.format(name))
                continue
            indices = torch.tensor(probe['indices'], device=gradient.device, dtype=torch.long)
            values = gradient.detach().flatten()[indices].float().cpu()
            if not torch.isfinite(values).all().item():
                raise RuntimeError('Nonfinite reference gradient: {}'.format(name))
            probe['values'] += values / world_size
        del gradients, loss, inputs
    model.zero_grad(set_to_none=True)
    for probe in probes.values():
        probe['values'] = probe['values'].tolist()
    return {
        'tokens': tokens,
        'loss': sum(losses) / world_size,
        'loss_per_rank': losses,
        'gradients': probes,
        'batch_size': batch_size,
        'seq_len': seq_len,
        'world_size': world_size,
        'reference_microbatch_size': batch_size,
    }


def inference_reference(model, requests):
    """Compare cached decode to full-prefix logits and engine-selected logprobs."""
    import torch

    if not isinstance(requests, list) or not requests:
        raise ValueError('Inference reference requests must be nonempty.')
    model.eval()
    comparisons = []
    with torch.no_grad():
        for request_index, request in enumerate(requests):
            prompt = request['prompt_token_ids']
            completion = request['token_ids']
            logprobs = request['logprobs']
            if not prompt or not completion or len(completion) != len(logprobs):
                raise ValueError('Incomplete inference reference request.')
            full_prefix_steps = request.get('full_prefix_steps', list(range(len(completion))))
            if (
                not isinstance(full_prefix_steps, list) or not full_prefix_steps
                or any(type(step) is not int
                       for step in full_prefix_steps) or full_prefix_steps != sorted(set(full_prefix_steps))
                or full_prefix_steps[0] != 0 or full_prefix_steps[-1] != len(completion) - 1
            ):
                raise ValueError('Full-prefix reference positions must include ordered first and final steps.')
            prefix = torch.tensor([prompt], device='cuda', dtype=torch.long)
            cache = None
            for step, token in enumerate(completion):
                attention_mask = torch.ones_like(prefix)
                output = model(
                    input_ids=prefix if step == 0 else prefix[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True
                )
                cache = output.past_key_values
                cached = output.logits[:, -1, :].float()
                check_full_prefix = step in full_prefix_steps
                full = (
                    model(input_ids=prefix, attention_mask=attention_mask, use_cache=False).logits[:, -1, :].float()
                    if check_full_prefix else cached
                )
                if not torch.isfinite(cached).all().item() or not torch.isfinite(full).all().item():
                    raise RuntimeError('Nonfinite reference logits.')
                if check_full_prefix:
                    torch.testing.assert_close(cached, full, rtol=0.02, atol=0.05)
                if full[0].argmax().item() != token:
                    raise RuntimeError('Engine greedy token differs at request {} step {}.'.format(request_index, step))
                expected = torch.log_softmax(full[0], dim=-1)
                ids = [int(value) for value in logprobs[step]]
                if token not in ids or not ids:
                    raise ValueError('Missing selected-token engine log probability.')
                actual = torch.tensor([logprobs[step][str(value)] for value in ids], device='cuda')
                target = expected[ids]
                torch.testing.assert_close(
                    actual,
                    target,
                    rtol=0.02,
                    atol=0.05,
                    msg=lambda message: 'Engine/reference request {} step {}:\n{}'.format(request_index, step, message),
                )
                comparisons.append(
                    {
                        'request': request_index,
                        'step': step,
                        'logprob_positions': len(ids),
                        'full_prefix_checked': check_full_prefix,
                        'cache_max_abs_error': (cached - full).abs().max().item() if check_full_prefix else None,
                        'engine_logprob_max_abs_error': (actual - target).abs().max().item(),
                    }
                )
                prefix = torch.cat((prefix, torch.tensor([[token]], device='cuda')), dim=1)
    return {'comparisons': comparisons, 'logprob_rtol': 0.02, 'logprob_atol': 0.05}


def main():
    """Run a bounded reference and write only numeric/token-ID evidence."""
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=['training', 'inference'], required=True)
    parser.add_argument('--requests')
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=32)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if args.mode == 'training' else torch.float32
    model = load_model(args.checkpoint, dtype).cuda()
    try:
        if args.mode == 'training':
            result = training_reference(model, args.checkpoint, args.batch_size, args.seq_len, args.world_size)
        else:
            if not args.requests:
                raise ValueError('--requests is required for inference reference.')
            result = inference_reference(model, read_json(Path(args.requests)))
        torch.cuda.synchronize()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    result['correctness_passed'] = True
    result['teardown_complete'] = True
    result['reference_dtype'] = str(dtype)
    result['tf32_allowed'] = False
    write_json(Path(args.output), result)


if __name__ == '__main__':
    main()
