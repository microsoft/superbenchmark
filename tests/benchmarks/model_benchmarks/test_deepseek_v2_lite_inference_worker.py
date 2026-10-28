# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for generation counters and the pinned engine timing contract."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
                logprobs=[
                    {key: SimpleNamespace(logprob=-0.5 if key == token else -2.0)
                     for key in set(range(20)) | {token}} for token in tokens
                ]
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
    assert result['logprobs'][0]['7'] == result['logprobs'][1]['8'] == -0.5
    assert all(len(row) == 20 for row in result['logprobs'])


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


def test_truncated_logprob_row_is_not_accepted(worker):
    """Numerical validation must retain the requested top-20 positions, not only the chosen token."""
    output = response()
    del output.outputs[0].logprobs[0][0]
    with pytest.raises(RuntimeError, match='top-20'):
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
    """Exclude distinct warmup timings and retain mixed EOS/multi-token measurements."""
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    clock = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
    monkeypatch.setattr(worker.time, 'monotonic', lambda: next(clock))
    batches = iter(
        [
            [(1, 0.05, 0.06, 0), (4, 0.075, 0.085, 0.015)],
            [(1, 0.01, 0.015, 0), (3, 0.02, 0.025, 0.12)],
            [(2, 0.03, 0.035, 0.08), (1, 0.04, 0.045, 0)],
        ]
    )
    submitted = []
    monkeypatch.setattr(worker.time, 'time', lambda: 1000.0 + 100 * len(submitted))

    class Engine:
        def generate(self, prompts, parameters, use_tqdm):
            submitted.append(prompts)
            outputs = []
            for prompt, (count, prefill, ttft, decode) in zip(prompts, next(batches)):
                output = response((99, ) if count == 1 else tuple(range(7, 7 + count)))
                output.prompt_token_ids = prompt['prompt_token_ids']
                output.metrics.first_token_ts = output.metrics.scheduled_ts + prefill
                output.metrics.first_token_latency = ttft
                output.metrics.last_token_ts = output.metrics.first_token_ts + decode
                if count == 1:
                    output.outputs[0].finish_reason = 'stop'
                outputs.append(output)
            return outputs

    args = SimpleNamespace(num_warmup=1, num_steps=2, seq_len=32, batch_size=2, max_new_tokens=4)
    report = worker.measured_generation(Engine(), [[1, 2, 3], [4, 5]], 99, args)
    assert len(submitted) == 3
    assert report['step_ms'] == pytest.approx([200, 300])
    assert report['request_count'] == 4
    assert report['input_token_counts'] == [32, 32, 32, 32]
    assert report['output_token_counts'] == [1, 3, 2, 1]
    assert report['request_prefill_ms'] == pytest.approx([10, 20, 30, 40])
    assert report['ttft_ms'] == pytest.approx([15, 25, 35, 45])
    assert report['decode_tpot_ms'] == pytest.approx([60, 80])
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


def test_numerical_checks_cover_real_profile_and_later_decode(worker, monkeypatch):
    """Use a full performance-shaped batch and retain complete selected-token traces."""
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    submitted = []

    class Engine:
        def generate(self, prompts, parameters, use_tqdm):
            submitted.append((prompts, parameters))
            outputs = []
            for prompt in prompts:
                output = response(tuple(range(64)))
                output.prompt_token_ids = prompt['prompt_token_ids']
                outputs.append(output)
            return outputs

    args = SimpleNamespace(seq_len=512, batch_size=8, max_new_tokens=64)
    records = worker.profile_checks(Engine(), [[1, 2, 3], [4, 5]], 99, args)
    prompts, parameters = submitted[0]
    assert len(prompts) == 8
    assert all(len(prompt['prompt_token_ids']) == 512 for prompt in prompts)
    assert parameters.max_tokens == 64 and parameters.logprobs == 20
    assert [record['profile_request_index'] for record in records] == [0, 7]
    assert all(record['profile_batch_size'] == 8 for record in records)
    assert all(len(record['token_ids']) == len(record['logprobs']) == 64 for record in records)
    assert all(record['full_prefix_steps'] == [0, 1, 32, 63] for record in records)


def test_numerical_profile_checks_reject_partial_batch(worker, monkeypatch):
    """Numerical sampling must not hide requests dropped from its full batch."""
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    engine = SimpleNamespace(generate=lambda *args, **kwargs: [])
    args = SimpleNamespace(seq_len=512, batch_size=8, max_new_tokens=64)
    with pytest.raises(RuntimeError, match='lost a request'):
        worker.profile_checks(engine, [[1, 2, 3]], 99, args)


@pytest.fixture
def main_driver(worker, tmp_path, monkeypatch):
    """Exercise the full worker's control flow without real vLLM or GPU execution."""
    if sys.version_info < (3, 10):
        pytest.skip('The standalone inference runtime requires Python 3.10 or later.')
    import importlib.metadata
    torch = pytest.importorskip('torch')

    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.31.0' if name == 'vllm' else version(name))
    for variable in [
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE', 'VLLM_FLOAT32_MATMUL_PRECISION',
        'TRITON_F32_DEFAULT', 'VLLM_RAISE_ON_LOGIT_NANS', 'VLLM_BATCH_INVARIANT', 'VLLM_MLA_DISABLE'
    ]:
        monkeypatch.setenv(variable, 'test-original')
    monkeypatch.setattr(worker.signal, 'signal', Mock())
    monkeypatch.setattr(worker, 'configure_cache', Mock())
    download = Mock(return_value=tmp_path / 'checkpoint')
    monkeypatch.setattr(worker, 'download_checkpoint', download)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'synchronize', Mock())
    monkeypatch.setattr(torch.cuda, 'empty_cache', Mock())
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'CPU orchestration fixture')
    shutdown = Mock()
    config = SimpleNamespace(dtype=torch.bfloat16, quantization=None)
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3], eos_token_id=99)

    def generate(prompts, parameters, use_tqdm):
        outputs = []
        stop = hasattr(parameters, 'allowed_token_ids')
        tokens = parameters.allowed_token_ids if stop else list(range(7, 7 + parameters.max_tokens))
        for prompt in prompts:
            output = response(tuple(tokens))
            output.prompt_token_ids = prompt['prompt_token_ids']
            if stop:
                output.outputs[0].finish_reason = 'stop'
                output.metrics.last_token_ts = output.metrics.first_token_ts
            outputs.append(output)
        return outputs

    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=config, engine_core=SimpleNamespace(shutdown=shutdown)),
        get_tokenizer=lambda: tokenizer,
        generate=generate,
    )
    factory = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=factory, SamplingParams=SimpleNamespace))

    def reference(command, log, timeout):
        assert timeout > 0
        assert shutdown.called
        output = Path(command[command.index('--output') + 1])
        worker.write_json(output, {'correctness_passed': True, 'teardown_complete': True})

    oracle = Mock(side_effect=reference)
    monkeypatch.setattr(worker, 'run_worker', oracle)
    output = tmp_path / 'report.json'
    argv = [
        'worker',
        '--cache_dir',
        str(tmp_path / 'cache'),
        '--output',
        str(output),
        '--reference_python',
        'reference-python',
        '--batch_size',
        '2',
        '--seq_len',
        '64',
        '--num_warmup',
        '1',
        '--num_steps',
        '2',
        '--max_new_tokens',
        '4',
        '--duration',
        '60',
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    return SimpleNamespace(
        output=output,
        argv=argv,
        factory=factory,
        engine=engine,
        config=config,
        shutdown=shutdown,
        oracle=oracle,
        download=download,
        torch=torch,
    )


def test_inference_main_keeps_precision_counts_and_reference_lifecycle(worker, main_driver):
    """Check the selected policies and report only after shutdown and reference success."""
    state = main_driver
    worker.main()
    options = state.factory.call_args[1]
    assert options['dtype'] == 'bfloat16' and options['quantization'] is None
    assert options['hf_overrides'] == {'moe_router_dtype': 'float32'}
    assert options['max_model_len'] == 68 and options['max_num_seqs'] == 2
    report = json.loads(state.output.read_text())
    assert report['status'] == 'success' and report['correctness_passed'] and report['teardown_complete']
    assert report['request_count'] == 4 and report['output_token_counts'] == [4] * 4
    assert report['router_dtype'] == 'float32' and report['float32_matmul_precision'] == 'highest'
    assert report['numerical_policy'] == 'strict_fp32' and report['batch_invariant'] is False
    state.shutdown.assert_called_once_with(timeout=10)
    state.oracle.assert_called_once()


def test_nan_detection_is_enabled_before_vllm_import(worker, main_driver, monkeypatch):
    """Reject a disabled inherited setting before the engine module is imported."""
    real_import = __import__
    engine_imports = []

    def checked_import(name, *args, **kwargs):
        if name == 'vllm':
            assert worker.os.environ['VLLM_RAISE_ON_LOGIT_NANS'] == '1'
            engine_imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr('builtins.__import__', checked_import)
    worker.main()
    assert engine_imports


def test_corrupted_measured_generation_cannot_publish_success(worker, main_driver, monkeypatch):
    """Corruption in unscored measured requests must prevent result publication."""
    generate = main_driver.engine.generate

    def corrupted_generation(*args, **kwargs):
        outputs = generate(*args, **kwargs)
        for output in outputs:
            output.metrics.is_corrupted = True
        return outputs

    monkeypatch.setattr(main_driver.engine, 'generate', corrupted_generation)
    with pytest.raises(RuntimeError, match='logits were corrupted'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    main_driver.oracle.assert_not_called()
    assert not main_driver.output.exists()


def test_engine_error_still_shuts_down(worker, main_driver, monkeypatch):
    """Generation failure cannot leave the engine alive or produce performance success."""
    monkeypatch.setattr(main_driver.engine, 'generate', Mock(side_effect=RuntimeError('generation failed')))
    with pytest.raises(RuntimeError, match='generation failed'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    main_driver.oracle.assert_not_called()
    assert not main_driver.output.exists()


def test_failed_reference_is_not_success(worker, main_driver):
    """A numerically rejected generation run must not publish a success report."""
    main_driver.oracle.side_effect = RuntimeError('numerical gate failed')
    with pytest.raises(RuntimeError, match='numerical gate'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    assert not main_driver.output.exists()


def test_wrong_engine_precision_is_rejected(worker, main_driver):
    """Do not silently replace the BF16 performance workload with another precision."""
    main_driver.config.dtype = main_driver.torch.float32
    with pytest.raises(RuntimeError, match='unquantized BF16'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    assert not main_driver.output.exists()


def test_unsupported_engine_version_fails_before_download(worker, main_driver, monkeypatch):
    """Version-specific timing and shutdown contracts must not silently drift."""
    import importlib.metadata
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.32.0' if name == 'vllm' else version(name))
    with pytest.raises(RuntimeError, match='pinned vLLM'):
        worker.main()
    main_driver.download.assert_not_called()


def test_inference_does_not_fall_back_to_cpu(worker, main_driver, monkeypatch):
    """Unavailable GPU execution is an explicit failure, not a replacement benchmark."""
    monkeypatch.setattr(main_driver.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='GPU unavailable'):
        worker.main()
    main_driver.download.assert_not_called()


def test_reference_success_code_requires_correctness_flags(worker, main_driver):
    """Returning from the oracle process is insufficient without its explicit checks."""
    def rejected(command, log, timeout):
        output = Path(command[command.index('--output') + 1])
        worker.write_json(output, {'correctness_passed': False, 'teardown_complete': True})

    main_driver.oracle.side_effect = rejected
    with pytest.raises(RuntimeError, match='reference did not complete'):
        worker.main()
    assert not main_driver.output.exists()


@pytest.mark.parametrize(
    'override', [
        ['--batch_size', '0'],
        ['--seq_len', '16'],
        ['--num_steps', '0'],
        ['--num_warmup', '-1'],
        ['--max_new_tokens', '3'],
        ['--gpu_memory_utilization', 'nan'],
        ['--duration', '3600'],
    ]
)
def test_inference_preflight_rejects_invalid_profile(worker, main_driver, monkeypatch, override):
    """Invalid generation parameters must fail before download or engine creation."""
    monkeypatch.setattr(sys, 'argv', main_driver.argv + override)
    with pytest.raises(ValueError):
        worker.main()
    main_driver.download.assert_not_called()
    main_driver.factory.assert_not_called()


def h100_arguments(worker, main_driver):
    """Select the frozen shape without changing the existing small fixture's defaults."""
    arguments = main_driver.argv + ['--numerical_policy', worker.numerics.POLICY]
    for key, value in worker.numerics.PROFILE.items():
        arguments.extend(['--' + key, str(value)])
    return arguments


def test_h100_checks_retain_every_window_and_batch_row(worker, main_driver):
    """The runtime replays all qualification windows, not only first/last rows."""
    args = SimpleNamespace(**worker.numerics.PROFILE)
    documents, eos = [[1, 2, 3], [4, 5]], 99
    records = worker.qualified_checks(main_driver.engine, documents, eos, args)
    assert len(records) == 84
    groups = worker.numerics.validate_requests(records, documents, eos)
    assert list(groups) == list(worker.numerics.OFFSETS)
    assert all(sum(len(record['token_ids']) for record in group) == 536 for group in groups.values())


def test_h100_policy_uses_pinned_engine_settings_and_separate_reference(worker, main_driver, monkeypatch):
    """Wire the explicit policy without replacing the measured BF16 model."""
    monkeypatch.setattr(sys, 'argv', h100_arguments(worker, main_driver))
    expected_gpu = worker.numerics.load_contract()['calibrated_contract']['runtime_identity']['gpu_name']
    monkeypatch.setattr(main_driver.torch.cuda, 'get_device_name', lambda index: expected_gpu)
    main_driver.config.use_mla = True
    validation = Mock()
    monkeypatch.setattr(worker.numerics, 'validate_reference_result', validation)
    worker.main()
    report = json.loads(main_driver.output.read_text())
    assert report['numerical_policy'] == worker.numerics.POLICY and report['batch_invariant'] is True
    assert main_driver.factory.call_args.kwargs['dtype'] == 'bfloat16'
    command = main_driver.oracle.call_args.args[0]
    assert command[1].endswith('deepseek_v2_lite_numerics.py') and '--mode' not in command
    assert len(json.loads((main_driver.output.parent / 'reference-requests.json').read_text())) == 84
    validation.assert_called_once_with(report['reference'])
    assert worker.os.environ['VLLM_BATCH_INVARIANT'] == '1'
    assert worker.os.environ['VLLM_MLA_DISABLE'] == '0'
    main_driver.shutdown.assert_called_once()


def test_h100_policy_rejects_another_gpu_before_download(worker, main_driver, monkeypatch):
    """A qualified H100 policy cannot be silently reused on other hardware."""
    monkeypatch.setattr(sys, 'argv', h100_arguments(worker, main_driver))
    with pytest.raises(ValueError, match='qualified H100 hardware'):
        worker.main()
    main_driver.download.assert_not_called()


def test_h100_policy_rejects_expanded_kv_and_still_shuts_down(worker, main_driver, monkeypatch):
    """Preserve the measured implementation as well as model dtype."""
    monkeypatch.setattr(sys, 'argv', h100_arguments(worker, main_driver))
    expected_gpu = worker.numerics.load_contract()['calibrated_contract']['runtime_identity']['gpu_name']
    monkeypatch.setattr(main_driver.torch.cuda, 'get_device_name', lambda index: expected_gpu)
    main_driver.config.use_mla = False
    with pytest.raises(RuntimeError, match='compressed-MLA'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    main_driver.oracle.assert_not_called()
