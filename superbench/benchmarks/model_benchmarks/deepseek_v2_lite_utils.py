# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Checkpoint, input and subprocess contracts for the fixed V2-Lite workload."""

import hashlib
import json
import math
import os
import signal
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

MODEL_ID = 'deepseek-ai/DeepSeek-V2-Lite'
REVISION = '604d5664dddd88a0433dbae533b7fe9472482de0'
PARAMETER_COUNT = 15706484224
WEIGHT_FILES = {
    'model-00001-of-000004.safetensors':
    (8594887408, '0d7e9f39bde40111a4c0f390b87497dce4565cf578d916395e6b2c7851f1e8da'),
    'model-00002-of-000004.safetensors':
    (8591757448, 'e0656832b0d594b4a64cad40ff8465231de6ed44c854f74f6b217797660aa4bb'),
    'model-00003-of-000004.safetensors':
    (8590718520, '843ec689624f3a520526e040f0326c4dc9865e8172942ca98a084fe136fdb21a'),
    'model-00004-of-000004.safetensors':
    (5636263200, 'cfb51658f67cedfbbc4d62ad14187830ceec8ee82c788c5f718feea98905ef31'),
}
WEIGHT_BYTES = sum(size for size, _ in WEIGHT_FILES.values())
EXPECTED_CONFIG = {
    'hidden_size': 2048,
    'num_hidden_layers': 27,
    'num_attention_heads': 16,
    'intermediate_size': 10944,
    'moe_intermediate_size': 1408,
    'n_routed_experts': 64,
    'n_shared_experts': 2,
    'num_experts_per_tok': 6,
    'first_k_dense_replace': 1,
    'kv_lora_rank': 512,
    'q_lora_rank': None,
    'qk_nope_head_dim': 128,
    'qk_rope_head_dim': 64,
    'v_head_dim': 128,
    'vocab_size': 102400,
}
FIXED_TEXTS = (
    'The engineering team measures a system before changing its configuration. '
    'Each measurement records the input, the software version, and the observation time.',
    'A warehouse receives several shipments each morning. Workers check the labels, '
    'sort the packages by destination, and keep a record of damaged items.',
    'A weather station samples temperature and wind speed at regular intervals. '
    'Missing readings are reported explicitly instead of being replaced with zero.',
    'The library contains books about science, history, and music. Readers can search '
    'the catalog, reserve an available book, and return it at the front desk.',
    'A numerical experiment compares two implementations on the same input. '
    'The result includes the error tolerance and every failed comparison.',
    'A small garden has three rows of vegetables and a rainwater tank. '
    'The gardener checks soil moisture before deciding how much water to use.',
    'The train arrives at the northern platform at eight in the morning. '
    'Passengers check their tickets and leave space for people getting off.',
    'A computer processes a sequence of tokens using a language model. '
    'Training computes a loss and gradients, while generation reuses attention state.',
)
INPUT_SHA256 = hashlib.sha256('\n'.join(FIXED_TEXTS).encode('utf-8')).hexdigest()


def read_json(path: Path):
    """Read a JSON document without accepting nonfinite JSON constants."""
    def reject_constant(value):
        raise ValueError('Nonfinite JSON constant: {}'.format(value))

    with path.open() as stream:
        return json.load(stream, parse_constant=reject_constant)


def write_json(path: Path, document) -> None:
    """Write a complete report atomically, rejecting nonfinite numbers."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def validate_config(config) -> None:
    """Reject a reduced, quantized or different model configuration."""
    if not isinstance(config, dict) or config.get('model_type') != 'deepseek_v2':
        raise ValueError('Expected the official DeepSeek-V2 configuration.')
    for key, expected in EXPECTED_CONFIG.items():
        if key not in config or config[key] != expected:
            raise ValueError('Official V2-Lite {} must be {!r}, got {!r}.'.format(key, expected, config.get(key)))
    if config.get('quantization_config') is not None:
        raise ValueError('This profile does not accept quantized checkpoints.')


def configure_cache(cache: Path) -> None:
    """Isolate all model caches without reading or removing shared caches."""
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.iterdir()):
        raise ValueError('Cold-start cache must be empty: {}'.format(cache))
    set_cache_environment(cache)


def set_cache_environment(cache: Path) -> None:
    """Select the same isolated cache in each distributed worker."""
    for variable, directory in {
        'HF_HOME': 'hf',
        'HF_HUB_CACHE': 'hub',
        'HF_XET_CACHE': 'xet',
        'HF_MODULES_CACHE': 'modules',
        'TRANSFORMERS_CACHE': 'transformers',
    }.items():
        os.environ[variable] = str(cache / directory)
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


def verify_checkpoint(checkpoint: Path) -> None:
    """Verify the complete fixed checkpoint, its index and every weight hash."""
    validate_config(read_json(checkpoint / 'config.json'))
    index = read_json(checkpoint / 'model.safetensors.index.json')
    if not isinstance(index, dict) or not isinstance(index.get('weight_map'), dict):
        raise ValueError('Missing checkpoint weight map.')
    if set(index['weight_map'].values()) != set(WEIGHT_FILES):
        raise ValueError('Checkpoint index does not describe the four official weight shards.')
    for filename, (size, expected_hash) in WEIGHT_FILES.items():
        path = checkpoint / filename
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError('Missing or incorrectly sized checkpoint shard: {}'.format(filename))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError('Checkpoint SHA256 mismatch: {}'.format(filename))


def download_checkpoint(cache: Path) -> Path:
    """Download and verify official weights before any GPU model construction."""
    from huggingface_hub import snapshot_download

    checkpoint = cache / 'checkpoint'
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(checkpoint),
        cache_dir=str(cache / 'hub'),
        allow_patterns=list(WEIGHT_FILES) + [
            'config.json', 'generation_config.json', 'model.safetensors.index.json',
            'tokenizer*', 'special_tokens_map.json', '*.py'
        ],
        token=False,
        max_workers=4,
    )
    verify_checkpoint(checkpoint)
    return checkpoint


def packed_tokens(
    documents: Sequence[Sequence[int]], eos: int, length: int, count: int, offset: int = 0
) -> List[List[int]]:
    """Pack deterministic real-text tokens without padding a pressure workload."""
    if length < 2 or count < 1 or offset < 0 or not documents or eos < 0:
        raise ValueError('Invalid packed-input dimensions or documents.')
    stream: List[int] = []
    for document in documents:
        if not document or any(type(token) is not int or token < 0 for token in document):
            raise ValueError('Each document must contain nonnegative integer token IDs.')
        stream.extend(document)
        stream.append(eos)
    return [
        [stream[(offset + row * length + column) % len(stream)] for column in range(length)]
        for row in range(count)
    ]


def numeric_series(report, name: str, count: int, positive: bool = True) -> List[float]:
    """Require a complete finite metric series, never a default or sentinel."""
    values = report.get(name) if isinstance(report, dict) else None
    if not isinstance(values, list) or len(values) != count:
        raise ValueError('Expected {} {} samples.'.format(count, name))
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        or (positive and value <= 0) for value in values
    ):
        raise ValueError('Invalid numeric metric: {}'.format(name))
    return [float(value) for value in values]


def uniform_indices(length: int, count: int = 256) -> List[int]:
    """Sample large tensor offsets with integer arithmetic, never FP32 rounding."""
    if length < 1 or count < 2:
        raise ValueError('Invalid tensor length or sample count.')
    return [position * (length - 1) // (count - 1) for position in range(count)]


def terminate_worker(process: subprocess.Popen) -> None:
    """Terminate only this job's process group, including torchrun workers."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.wait()
        return
    process.wait(timeout=5)


def worker_group_exists(pid: int) -> bool:
    """Check the specific job group, including children of an exited leader."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    return True


def interrupt_worker(signum, frame) -> None:
    """Unwind subprocess waits so nested reference workers are cleaned up."""
    raise TimeoutError('Model worker interrupted by signal {}.'.format(signum))


def run_worker(command: List[str], log: Path, timeout: float) -> None:
    """Run an isolated process group under one deadline, preserving failure logs."""
    if timeout <= 0:
        raise TimeoutError('No time remains for the worker.')
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('Model worker exceeded the download-inclusive deadline.') from error
            if code != 0:
                raise RuntimeError('Model worker exited with code {}; see {}.'.format(code, log))
            if worker_group_exists(process.pid):
                raise RuntimeError('Model worker left its subprocess group running; see {}.'.format(log))
        finally:
            if worker_group_exists(process.pid):
                terminate_worker(process)


def identity_report() -> Dict[str, object]:
    """Describe the fixed model and input contract without generated text."""
    return {
        'model_id': MODEL_ID,
        'revision': REVISION,
        'weight_bytes': WEIGHT_BYTES,
        'weight_sha256': {name: digest for name, (_, digest) in WEIGHT_FILES.items()},
        'parameter_count': PARAMETER_COUNT,
        'precision': 'bfloat16',
        'input_dataset': 'fixed-text-v1',
        'input_sha256': INPUT_SHA256,
        'cold_cache': True,
    }


def validate_identity(report) -> None:
    """Require a successful official-model report before accepting any metrics."""
    expected = identity_report()
    if not isinstance(report, dict) or report.get('status') != 'success':
        raise ValueError('Worker did not produce a successful model report.')
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('Worker report does not match the official checkpoint/input contract.')
    if report.get('correctness_passed') is not True or report.get('teardown_complete') is not True:
        raise ValueError('Correctness or worker teardown was not completed.')
