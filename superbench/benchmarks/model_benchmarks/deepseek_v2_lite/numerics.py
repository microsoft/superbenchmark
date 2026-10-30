# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Independent input-matched BF16 and full-prefix FP32 inference references."""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

from .utils import packed_tokens, read_json, write_json

METHOD = 'bf16-input-matched-probability-log-v6'
POLICY = 'h100_bf16_v6'
POLICIES = ('strict_fp32', POLICY)
PROFILE = {'batch_size': 8, 'seq_len': 512, 'max_new_tokens': 64, 'num_warmup': 2, 'num_steps': 256}
OFFSETS = (0, 1024, 32768, 98304, 130560, 131584)
CONTRACT_SHA256 = '1f95fe705da2bf8cba1b7846b8a84592add397d6bfc94acf691dbed7756de008'
TOP_K = 20
SLACK = 1e-6
METRICS = ('max_probability_error', 'relative_probability_mass_error', 'normalized_logprob_error')
LIMIT_RULE = 'reference calibration budget + input-matched BF16 reference deviation'


def canonical_hash(value):
    """Hash a policy independently of JSON whitespace."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_profile(args):
    """Keep the original policy and explicitly restrict the qualified H100 shape."""
    if args.numerical_policy not in POLICIES:
        raise ValueError('Unknown V2-Lite numerical policy.')
    if args.numerical_policy == POLICY and any(getattr(args, key) != value for key, value in PROFILE.items()):
        raise ValueError('The H100 numerical policy requires its complete frozen generation profile.')


def load_contract():
    """Require the shipped, qualified artifact rather than caller-supplied tolerances."""
    if not CONTRACT_SHA256:
        raise RuntimeError('The H100 numerical policy has not been installed as a qualified contract.')
    path = Path(__file__).with_name('h100_bf16_v6.json')
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != CONTRACT_SHA256:
        raise ValueError('The qualified H100 numerical contract changed.')
    contract = json.loads(content)
    candidate = contract['calibrated_contract']
    payload = {key: value for key, value in candidate.items() if key != 'frozen_limits_sha256'}
    if (
        contract.get('status') != 'qualified' or candidate.get('method') != METHOD
        or candidate.get('reference_only') is not True or candidate.get('input_matched_limit_rule') != LIMIT_RULE
        or canonical_hash(payload) != candidate.get('frozen_limits_sha256')
        or candidate.get('qualification_offsets') != list(OFFSETS)
        or candidate.get('profile') != {key: PROFILE[key]
                                        for key in ('batch_size', 'seq_len', 'max_new_tokens')}
        or candidate.get('qualification_stream') != {key: PROFILE[key]
                                                     for key in ('num_warmup', 'num_steps')}
        or {row['profile_offset']
            for row in contract.get('qualification', [])} != set(OFFSETS)
        or contract.get('reference_validation', {}).get('non_anchor_comparisons', 0) <= 0
    ):
        raise ValueError('Incomplete or incompatible H100 numerical qualification.')
    return contract


def probability_errors(reference_logprobs, observed_logprobs):
    """Compare every represented token and the unrepresented probability mass."""
    if not reference_logprobs or len(reference_logprobs) != len(observed_logprobs):
        raise ValueError('Probability comparisons must be nonempty and aligned.')
    for values in (reference_logprobs, observed_logprobs):
        if any(type(value) not in (float, int) or not math.isfinite(value) or value > 0 for value in values):
            raise ValueError('Log probabilities must be finite and nonpositive.')
    expected = [math.exp(value) for value in reference_logprobs]
    actual = [math.exp(value) for value in observed_logprobs]
    expected_mass, actual_mass = math.fsum(expected), math.fsum(actual)
    if expected_mass <= 0 or max(expected_mass, actual_mass) > 1 + SLACK:
        raise ValueError('Invalid represented probability mass.')
    differences = [abs(left - right) for left, right in zip(expected, actual)]
    variation = 0.5 * (math.fsum(differences) + abs(expected_mass - actual_mass))
    return {
        'max_probability_error':
        max(differences),
        'coarsened_total_variation':
        variation,
        'relative_probability_mass_error':
        variation / expected_mass,
        'reference_mass':
        expected_mass,
        'observed_mass':
        actual_mass,
        'max_logprob_error':
        max(abs(left - right) for left, right in zip(reference_logprobs, observed_logprobs)),
        'normalized_logprob_error':
        max(
            abs(left - right) / (0.05 + 0.02 * abs(left)) for left, right in zip(reference_logprobs, observed_logprobs)
        ),
    }


def tensor_errors(expected, actual):
    """Measure reference variation on both natural top-k supports."""
    results = [
        probability_errors(expected[support].tolist(), actual[support].tolist())
        for support in (expected.topk(TOP_K).indices, actual.topk(TOP_K).indices)
    ]
    return {
        'max_probability_error': (expected.double().exp() - actual.double().exp()).abs().max().item(),
        'relative_probability_mass_error': max(row['relative_probability_mass_error'] for row in results),
        'coarsened_total_variation': max(row['coarsened_total_variation'] for row in results),
        'normalized_logprob_error': max(row['normalized_logprob_error'] for row in results),
        'greedy_match': expected.argmax().item() == actual.argmax().item(),
    }


def input_matched_limits(calibrated, reference_errors):
    """Never use engine probabilities to determine their own acceptance budget."""
    limits = {}
    for metric in METRICS:
        budget, deviation = calibrated[metric], reference_errors[metric]
        if (
            type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0
            or type(deviation) not in (int, float) or not math.isfinite(deviation) or deviation < 0
        ):
            raise ValueError('Invalid calibrated budget or input-matched reference error.')
        limits[metric] = budget + deviation
    if limits['relative_probability_mass_error'] >= 0.5:
        raise ValueError('Input-matched reference uncertainty cannot distinguish half-normalization.')
    return limits


def score_probabilities(expected, observed, chosen_token, reference_best, limits):
    """Require all three numerical metrics and a reference-bounded greedy choice."""
    if set(expected) != set(observed) or chosen_token not in observed or reference_best not in observed:
        raise ValueError('Incomplete aligned probabilities or missing reference/engine greedy token.')
    ids = sorted(expected)
    metrics = probability_errors([expected[key] for key in ids], [observed[key] for key in ids])
    reasons = []
    for name in METRICS:
        bound = limits[name]
        if type(bound) not in (int, float) or not math.isfinite(bound) or bound <= 0:
            raise ValueError('Invalid frozen numerical bound.')
        if metrics[name] > bound:
            reasons.append(name)
    if observed[chosen_token] + SLACK < max(observed.values()):
        reasons.append('reported token is not greedy')
    gap = math.exp(expected[reference_best]) - math.exp(expected[chosen_token])
    effective_bound = min(
        limits['max_probability_error'], limits['relative_probability_mass_error'] * metrics['reference_mass']
    )
    if gap > 2 * effective_bound + SLACK:
        reasons.append('greedy choice exceeds calibrated uncertainty')
    return dict(
        metrics,
        accepted=not reasons,
        failures=reasons,
        greedy_match=chosen_token == reference_best,
        greedy_probability_gap=gap,
        greedy_gap_bound=2 * effective_bound,
    )


def evaluate_logits(reference_logprobs, observed, chosen_token, limits):
    """Do not reduce a top-20 check to its selected token."""
    if not 20 <= len(observed) <= 21:
        raise ValueError('Holdout records must retain the complete top-20 probabilities.')
    ids = [int(key) for key in observed]
    if len(set(ids)) != len(ids) or any(key < 0 or key >= reference_logprobs.numel() for key in ids):
        raise ValueError('A reported vocabulary position is duplicated or out of range.')
    expected = {key: reference_logprobs[key].item() for key in ids}
    actual = {int(key): value for key, value in observed.items()}
    reference_best = reference_logprobs.argmax().item()
    if reference_best not in actual:
        return {'accepted': False, 'failures': ['reference greedy token absent from complete engine top-k']}
    return score_probabilities(expected, actual, chosen_token, reference_best, limits)


def full_logits(model, token_rows):
    """Use a fresh complete prefix for every FP32 oracle position."""
    import torch

    inputs = torch.tensor(token_rows, device='cuda', dtype=torch.long)
    output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    logits = output.logits[:, -1, :].float().clone()
    del output
    if not torch.isfinite(logits).all().item():
        raise RuntimeError('Nonfinite official reference logits.')
    return logits


def cached_logits(model, cases):
    """Retain the complete batch's teacher-forced KV-cache trajectory."""
    import torch

    cache = None
    length = len(cases[0]['prompt_token_ids'])
    count = len(cases[0]['continuation_ids'])
    if any(len(case['prompt_token_ids']) != length or len(case['continuation_ids']) != count for case in cases):
        raise ValueError('Reference calibration batches must have matching sequence shapes.')
    for step in range(count):
        tokens = [case['prompt_token_ids'] if step == 0 else [case['continuation_ids'][step - 1]] for case in cases]
        inputs = torch.tensor(tokens, device='cuda', dtype=torch.long)
        mask = torch.ones((len(cases), length + step), device='cuda', dtype=torch.long)
        output = model(input_ids=inputs, attention_mask=mask, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float().clone()
        del output
        if not torch.isfinite(logits).all().item():
            raise RuntimeError('Nonfinite cached reference logits.')
        yield step, logits


def configured_batches(cases, batch_size):
    """Use the frozen configured shape and deterministic padding of incomplete groups."""
    if not cases or len({case['id'] for case in cases}) != len(cases) or batch_size < 1:
        raise ValueError('Reference batches require nonempty uniquely identified cases and a positive batch size.')
    batches = []
    for length, count in sorted({(len(case['prompt_token_ids']), len(case['continuation_ids'])) for case in cases}):
        selected = [
            case for case in cases if (len(case['prompt_token_ids']), len(case['continuation_ids'])) == (length, count)
        ]
        for start in range(0, len(selected), batch_size):
            batches.append([selected[(start + index) % len(selected)] for index in range(batch_size)])
    return batches


def collect_matched_bf16(model, cases, batch_size):
    """Compute reference values without access to any engine probabilities."""
    import torch

    values = {}
    with torch.no_grad():
        for batch in configured_batches(cases, batch_size):
            records = [[] for _ in batch]
            for step, logits in cached_logits(model, batch):
                if step != len(records[0]):
                    raise ValueError('Input-matched reference produced out-of-order decode positions.')
                for index, row in enumerate(logits.log_softmax(-1).cpu()):
                    records[index].append(row.clone())
            for case, rows in zip(batch, records):
                if len(rows) != len(case['continuation_ids']):
                    raise ValueError('Incomplete input-matched reference continuation.')
                if case['id'] not in values:
                    values[case['id']] = torch.stack(rows)
            print('Input-matched BF16 completed ' + batch[0]['id'], flush=True)
    if set(values) != {case['id'] for case in cases}:
        raise ValueError('Input-matched reference omitted a case.')
    return values


def validate_requests(requests, documents, eos):
    """Require every frozen window, complete configured batches and both short execution modes."""
    if not isinstance(requests, list) or len(requests) != len(OFFSETS) * (6 + PROFILE['batch_size']):
        raise ValueError('Incomplete H100 numerical request sets.')
    short = [packed_tokens(documents, eos, length, 1, index * 7)[0] for index, length in enumerate((16, 23, 32))]
    groups = {}
    for offset in OFFSETS:
        selected = [request for request in requests if request.get('numerical_set_offset') == offset]
        expected = short * 2 + packed_tokens(documents, eos, PROFILE['seq_len'], PROFILE['batch_size'], offset)
        if len(selected) != len(expected) or [request['prompt_token_ids'] for request in selected] != expected:
            raise ValueError('Numerical requests do not match the complete frozen input stream.')
        for index, request in enumerate(selected):
            maximum = 4 if index < 6 else PROFILE['max_new_tokens']
            tokens = request['token_ids']
            if (
                not isinstance(tokens, list) or not 0 < len(tokens) <= maximum
                or any(type(token) is not int or token < 0 for token in tokens)
                or len(request['logprobs']) != len(tokens)
            ):
                raise ValueError('Incomplete or invalid numerical continuation.')
            if index >= 6 and (
                request.get('profile_request_index') != index - 6
                or request.get('profile_batch_size') != PROFILE['batch_size']
            ):
                raise ValueError('Numerical evidence lost a configured-batch row.')
        if not any(len(request['token_ids']) == PROFILE['max_new_tokens'] for request in selected[6:]):
            raise ValueError('Numerical window did not exercise the complete decode length.')
        groups[offset] = selected
    return groups


def _validate_window_result(window):
    """Check all request positions and corresponding faults in one numerical window."""
    if not isinstance(window, dict):
        raise ValueError('Invalid numerical window report.')
    positives, negatives = window.get('all_positive_records'), window.get('all_negative_records')
    if (
        not isinstance(positives, list) or not positives or not isinstance(negatives, list)
        or window.get('positive_positions') != len(positives)
        or window.get('negative_positions') != 2 * len(positives) + 1 or len(negatives) != window['negative_positions']
        or window.get('positive_failures') != [] or window.get('negative_failures') != []
        or any(not isinstance(row, dict) or row.get('accepted') is not True for row in positives)
        or any(not isinstance(row, dict) or row.get('accepted') is not False for row in negatives)
    ):
        raise ValueError('H100 numerical reference has missing or failed positions or faults.')
    if any(type(row.get(key)) is not int or row[key] < 0 for row in positives for key in ('request', 'step')):
        raise ValueError('Invalid numerical request or step indices.')
    positions = {(row['request'], row['step']) for row in positives}
    if len(positions) != len(positives) or {request for request, _ in positions} != set(range(14)):
        raise ValueError('H100 numerical reference lost or duplicated a request position.')
    full_decode = False
    for request in range(14):
        steps = {step for index, step in positions if index == request}
        maximum = 4 if request < 6 else PROFILE['max_new_tokens']
        if not 0 < len(steps) <= maximum or steps != set(range(len(steps))):
            raise ValueError('H100 numerical reference lost a decode position.')
        full_decode |= request >= 6 and len(steps) == PROFILE['max_new_tokens']
    expected_faults = {
        (fault, request, step)
        for request, step in positions for fault in ('half-normalization', 'temperature-two')
    } | {('last-request-context-for-first', 0, 0)}
    actual_faults = {(row.get('fault'), row.get('request'), row.get('step')) for row in negatives}
    if not full_decode or actual_faults != expected_faults or len(actual_faults) != len(negatives):
        raise ValueError('H100 numerical reference omitted complete decode or a required fault.')
    return len(positives), len(negatives)


def validate_reference_result(report):
    """Reject incomplete or substituted serialized reference evidence."""
    if not isinstance(report, dict) or (
        report.get('correctness_passed') is not True or report.get('teardown_complete') is not True
        or report.get('numerical_policy') != POLICY or report.get('contract_sha256') != CONTRACT_SHA256
        or report.get('profile_offsets') != list(OFFSETS) or report.get('request_count') != len(OFFSETS) *
        (6 + PROFILE['batch_size'])
    ):
        raise ValueError('Incomplete or mismatched H100 numerical reference result.')
    windows = report.get('windows')
    if not isinstance(windows, dict) or set(windows) != {str(offset) for offset in OFFSETS}:
        raise ValueError('H100 numerical reference omitted a frozen window.')
    positive_count, negative_count = 0, 0
    for window in windows.values():
        positives, negatives = _validate_window_result(window)
        positive_count += positives
        negative_count += negatives
    if report.get('positive_positions') != positive_count or report.get('negative_positions') != negative_count:
        raise ValueError('H100 numerical reference totals differ from its individual windows.')


def evaluate_group(model, requests, matched, limits):
    """Keep every positive and all faults, using one reference-only budget per position."""
    import torch

    positives, negatives = [], []
    contexts = {}
    with torch.no_grad():
        for request_index, request in enumerate(requests):
            for step, chosen in enumerate(request['token_ids']):
                prefix = request['prompt_token_ids'] + request['token_ids'][:step]
                expected = full_logits(model, [prefix])[0].log_softmax(-1).cpu()
                reference_errors = tensor_errors(expected, matched['request-{}'.format(request_index)][step])
                effective = input_matched_limits(limits, reference_errors)
                observed = request['logprobs'][step]
                result = evaluate_logits(expected, observed, chosen, effective)
                positives.append(
                    dict(
                        result,
                        request=request_index,
                        step=step,
                        input_reference_errors=reference_errors,
                        effective_limits=effective
                    )
                )
                half = {key: value - math.log(2) for key, value in observed.items()}
                result = evaluate_logits(expected, half, chosen, effective)
                negatives.append(dict(result, fault='half-normalization', request=request_index, step=step))
                wrong = (expected / 2).log_softmax(-1)
                ids = wrong.topk(TOP_K).indices.tolist()
                result = evaluate_logits(
                    expected, {str(key): wrong[key].item()
                               for key in ids},
                    wrong.argmax().item(), effective
                )
                negatives.append(dict(result, fault='temperature-two', request=request_index, step=step))
                if step == 0 and request_index in (0, len(requests) - 1):
                    contexts[request_index] = expected
            print('Full-prefix FP32 completed request {}'.format(request_index), flush=True)
    first, last = contexts[0], contexts[len(requests) - 1]
    ids = last.topk(TOP_K).indices.tolist()
    result = evaluate_logits(
        first, {str(key): last[key].item()
                for key in ids},
        last.argmax().item(), positives[0]['effective_limits']
    )
    negatives.append(dict(result, fault='last-request-context-for-first', request=0, step=0))
    return {
        'positive_positions': len(positives),
        'negative_positions': len(negatives),
        'positive_failures': [row for row in positives if not row['accepted']],
        'negative_failures': [row for row in negatives if row['accepted']],
        'all_positive_records': positives,
        'all_negative_records': negatives,
    }


def run_reference(checkpoint, requests, contract):
    """Keep measured BF16 inference separate from both independent reference models."""
    import torch
    from importlib.metadata import version
    from .reference import fixed_documents, load_model

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    candidate = contract['calibrated_contract']
    identity = {
        'torch_version': torch.__version__,
        'transformers_version': version('transformers'),
        'gpu_name': torch.cuda.get_device_name(0),
        'reference_loader_sha256': hashlib.sha256(inspect.getsource(load_model).encode()).hexdigest(),
        'tf32_allowed': False,
        'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
    if identity != candidate['runtime_identity']:
        raise ValueError('Reference runtime or hardware differs from the qualified H100 policy.')
    documents, eos = fixed_documents(str(checkpoint))
    groups = validate_requests(requests, documents, eos)
    model = load_model(str(checkpoint), torch.bfloat16).cuda().eval()
    matched = {}
    try:
        for offset, selected in groups.items():
            cases = [
                {
                    'id': 'request-{}'.format(index),
                    'prompt_token_ids': request['prompt_token_ids'],
                    'continuation_ids': request['token_ids']
                } for index, request in enumerate(selected)
            ]
            matched[offset] = collect_matched_bf16(model, cases, PROFILE['batch_size'])
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    results = {}
    model = load_model(str(checkpoint), torch.float32).cuda().eval()
    try:
        for offset, selected in groups.items():
            results[str(offset)] = evaluate_group(model, selected, matched[offset], candidate['limits'])
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    passed = all(not row['positive_failures'] and not row['negative_failures'] for row in results.values())
    return {
        'numerical_policy': POLICY,
        'contract_sha256': CONTRACT_SHA256,
        'frozen_limits_sha256': candidate['frozen_limits_sha256'],
        'runtime_identity': identity,
        'profile_offsets': list(OFFSETS),
        'request_count': len(requests),
        'positive_positions': sum(row['positive_positions'] for row in results.values()),
        'negative_positions': sum(row['negative_positions'] for row in results.values()),
        'correctness_passed': passed,
        'teardown_complete': True,
        'windows': results,
    }


def main():
    """Publish complete numerical evidence and fail the process on any rejected position or fault."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    result = run_reference(args.checkpoint, read_json(args.requests), load_contract())
    if result['correctness_passed'] is True:
        validate_reference_result(result)
    write_json(args.output, result)
    if result['correctness_passed'] is not True:
        raise RuntimeError('Input-matched H100 numerical validation rejected a position or accepted a fault.')


if __name__ == '__main__':
    main()
