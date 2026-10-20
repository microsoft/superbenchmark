# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for generation counters and the pinned engine timing contract."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def worker(monkeypatch):
    """Load the standalone worker without requiring a vLLM or CUDA installation."""
    directory = Path(__file__).resolve().parents[3] / 'superbench/benchmarks/model_benchmarks'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        'v2lite_generation_test', directory / 'deepseek_v2_lite_inference_worker.py'
    )
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the standalone generation worker.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def response(tokens=(7, 8)):
    """Construct engine-response fixtures without simulating GPU success."""
    return SimpleNamespace(
        finished=True,
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                token_ids=list(tokens),
                finish_reason='length',
                logprobs=[{
                    token: SimpleNamespace(logprob=-0.5)
                } for token in tokens]
            )
        ],
        metrics=SimpleNamespace(
            is_corrupted=False,
            num_generation_tokens=len(tokens),
            scheduled_ts=10.0,
            first_token_ts=10.25,
            last_token_ts=10.5,
            first_token_latency=0.4
        ),
    )


def test_actual_output_tokens_not_maximum(worker):
    """Count emitted token IDs instead of treating max_tokens as completed work."""
    result = worker.output_record(response(), 64, [1, 2, 3], True)
    assert result['token_ids'] == [7, 8]
    assert len(result['token_ids']) == 2
    assert result['logprobs'] == [{'7': -0.5}, {'8': -0.5}]


def test_eos_is_a_real_token(worker):
    """An EOS-only request has one output token and no decode denominator."""
    output = response((100001, ))
    output.outputs[0].finish_reason = 'stop'
    output.metrics.last_token_ts = output.metrics.first_token_ts
    assert worker.output_record(output, 64, [1, 2, 3])['token_ids'] == [100001]
    assert worker.request_times(output) == (250.0, 400.0, None)


def test_pinned_engine_timestamps(worker):
    """Derive durations from compatible monotonic timestamps and actual tokens."""
    assert worker.request_times(response()) == (250.0, 400.0, 250.0)


@pytest.mark.parametrize('tokens', [(), (1, 2, 3, 4, 5)])
def test_invalid_token_counts(worker, tokens):
    """Reject empty or over-limit outputs rather than inventing throughput."""
    with pytest.raises(RuntimeError, match='count'):
        worker.output_record(response(tokens), 4, [1, 2, 3])


@pytest.mark.parametrize('field,value', [('finished', False), ('prompt_token_ids', [])])
def test_incomplete_requests(worker, field, value):
    """Failed request batches do not count as complete model execution."""
    output = response()
    setattr(output, field, value)
    with pytest.raises(RuntimeError, match='unfinished'):
        worker.output_record(output, 64, [1, 2, 3])


def test_output_must_match_submitted_prompt(worker):
    """Do not validate a correct response against a different request's tokens."""
    with pytest.raises(RuntimeError, match='different prompt'):
        worker.output_record(response(), 64, [3, 2, 1])


def test_missing_logprobs(worker):
    """Missing numerical-reference evidence must fail explicitly."""
    output = response()
    output.outputs[0].logprobs = None
    with pytest.raises(RuntimeError, match='probabilities'):
        worker.output_record(output, 64, [1, 2, 3], True)


@pytest.mark.parametrize(
    'field,value', [
        ('is_corrupted', True), ('num_generation_tokens', 64), ('first_token_ts', 0.0),
        ('first_token_latency', float('nan')), ('last_token_ts', 10.0)
    ]
)
def test_invalid_statistics(worker, field, value):
    """Never substitute zero/default durations or hide corrupted logits."""
    output = response()
    setattr(output.metrics, field, value)
    with pytest.raises(RuntimeError):
        worker.request_times(output)


def test_measured_generation_excludes_warmup_and_counts_actual_tokens(worker, monkeypatch):
    """Aggregate only measured requests, retaining EOS counts without a false TPOT."""
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    clock = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
    monkeypatch.setattr(worker.time, 'monotonic', lambda: next(clock))
    lengths = iter([(4, 4), (1, 3), (2, 1)])
    submitted = []
    monkeypatch.setattr(worker.time, 'time', lambda: 1000.0 + 100 * len(submitted))

    class Engine:
        def generate(self, prompts, parameters, use_tqdm):
            submitted.append(prompts)
            outputs = []
            for prompt, count in zip(prompts, next(lengths)):
                output = response(tuple(range(7, 7 + count)))
                output.prompt_token_ids = prompt['prompt_token_ids']
                if count == 1:
                    output.outputs[0].finish_reason = 'stop'
                    output.metrics.last_token_ts = output.metrics.first_token_ts
                outputs.append(output)
            return outputs

    args = SimpleNamespace(num_warmup=1, num_steps=2, seq_len=32, batch_size=2, max_new_tokens=4)
    report = worker.measured_generation(Engine(), [[1, 2, 3], [4, 5]], 99, args)
    assert len(submitted) == 3
    assert report['step_ms'] == pytest.approx([200, 300])
    assert report['request_count'] == 4
    assert report['input_token_counts'] == [32, 32, 32, 32]
    assert report['output_token_counts'] == [1, 3, 2, 1]
    assert report['request_prefill_ms'] == [250, 250, 250, 250]
    assert report['ttft_ms'] == [400, 400, 400, 400]
    assert report['decode_tpot_ms'] == [125, 250]
    assert report['measurement_start_unix_seconds'] == 1100
    assert report['measurement_end_unix_seconds'] == 1300


def test_measured_generation_rejects_missing_requests(worker, monkeypatch):
    """A partial generation batch cannot become a valid throughput sample."""
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    engine = SimpleNamespace(generate=lambda *args, **kwargs: [])
    args = SimpleNamespace(num_warmup=0, num_steps=1, seq_len=32, batch_size=2, max_new_tokens=4)
    with pytest.raises(RuntimeError, match='fewer requests'):
        worker.measured_generation(engine, [[1, 2, 3]], 99, args)
