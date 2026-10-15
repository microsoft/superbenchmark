# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unsharded official-code reference, executed in a separate pinned environment."""

import argparse
import gc
from importlib.metadata import version
from pathlib import Path

from deepseek_v2_lite_utils import (
    FIXED_TEXTS, PARAMETER_COUNT, packed_tokens, read_json, uniform_indices, validate_config, write_json
)


def load_model(checkpoint, dtype):
    """Load official causal-LM weights with no quantization or model reduction."""
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


def training_reference(model, checkpoint):
    """Capture full-model loss and bounded gradient probes before optimization."""
    import torch

    documents, eos = fixed_documents(checkpoint)
    tokens = packed_tokens(documents, eos, 32, 1)
    inputs = torch.tensor(tokens, device='cuda', dtype=torch.long)
    model.train()
    output = model(input_ids=inputs, labels=inputs, use_cache=False)
    if output.loss is None or not torch.isfinite(output.loss).item():
        raise RuntimeError('Nonfinite or missing reference causal-LM loss.')
    output.loss.backward()
    required = {
        'model.embed_tokens.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.self_attn.kv_a_proj_with_mqa.weight',
        'model.layers.0.mlp.down_proj.weight',
        'model.layers.1.mlp.gate.weight',
    }
    probes = {}
    expert_probe = False
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
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
                'shape': list(parameter.shape), 'indices': indices.cpu().tolist(), 'values': values.tolist()
            }
            if selected_expert:
                expert_probe = True
    if not required.issubset(probes) or not expert_probe:
        raise RuntimeError('Required dense/MLA/router/expert reference gradients are missing.')
    return {'tokens': tokens, 'loss': output.loss.item(), 'gradients': probes}


def inference_reference(model, requests):
    """Compare cached decode to full-prefix logits and engine-selected logprobs."""
    import torch

    if not isinstance(requests, list) or not requests:
        raise ValueError('Inference reference requests must be nonempty.')
    model.eval()
    comparisons = []
    with torch.no_grad():
        for request in requests:
            prompt = request['prompt_token_ids']
            completion = request['token_ids']
            logprobs = request['logprobs']
            if not prompt or not completion or len(completion) != len(logprobs):
                raise ValueError('Incomplete inference reference request.')
            prefix = torch.tensor([prompt], device='cuda', dtype=torch.long)
            cache = None
            for step, token in enumerate(completion):
                output = model(input_ids=prefix if step == 0 else prefix[:, -1:], past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                cached = output.logits[:, -1, :].float()
                full = model(input_ids=prefix, use_cache=False).logits[:, -1, :].float()
                if not torch.isfinite(cached).all().item() or not torch.isfinite(full).all().item():
                    raise RuntimeError('Nonfinite reference logits.')
                torch.testing.assert_close(cached, full, rtol=0.02, atol=0.05)
                expected = torch.log_softmax(full[0], dim=-1)
                ids = [int(value) for value in logprobs[step]]
                if token not in ids or not ids:
                    raise ValueError('Missing selected-token engine log probability.')
                actual = torch.tensor([logprobs[step][str(value)] for value in ids], device='cuda')
                target = expected[ids]
                torch.testing.assert_close(actual, target, rtol=0.02, atol=0.05)
                comparisons.append({
                    'step': step,
                    'logprob_positions': len(ids),
                    'cache_max_abs_error': (cached - full).abs().max().item(),
                    'engine_logprob_max_abs_error': (actual - target).abs().max().item(),
                })
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
    args = parser.parse_args()
    torch.cuda.set_device(0)
    model = load_model(args.checkpoint, torch.bfloat16).cuda()
    try:
        if args.mode == 'training':
            result = training_reference(model, args.checkpoint)
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
    write_json(Path(args.output), result)


if __name__ == '__main__':
    main()
