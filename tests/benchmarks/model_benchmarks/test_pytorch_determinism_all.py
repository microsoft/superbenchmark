# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unified test for deterministic fingerprinting across all major PyTorch model benchmarks."""

import os
import tempfile
import json
import pytest
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, ReturnCode


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def run_deterministic_benchmark(model_name, params, log_path=None, extra_args=None):
    """
    Helper to launch a deterministic benchmark and return the result.
    """
    if log_path is None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
            log_path = tmpfile.name
    parameters = params + " --deterministic --deterministic_seed 42"
    if extra_args:
        parameters += " " + extra_args
    if "--generate-log" not in parameters:
        parameters += f" --generate-log --log-path {log_path} --check_frequency 10"
    context = BenchmarkRegistry.create_benchmark_context(
        model_name,
        platform=Platform.CUDA,
        parameters=parameters,
        framework=Framework.PYTORCH,
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    return benchmark, log_path


MODELS = [
    (
        "resnet18",
        "--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 --model_action train inference",
    ),
    (
        "lstm",
        "--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
    --model_action train inference --precision float32",
    ),
    (
        "gpt2-large",
        "--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 --model_action train inference",
    ),
    (
        "llama2-7b",
        "--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 --model_action train inference",
    ),
    (
        "mixtral-8x7b",
        "--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 "
        "--hidden_size 1024 --max_position_embeddings 2048 "
        "--intermediate_size 3584 --model_action train inference",
    ),
    (
        "bert-large",
        "--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 "
        "--num_steps 4 --model_action train inference",
    ),
]


@pytest.mark.parametrize("model_name, params", MODELS)
def test_pytorch_model_determinism(model_name, params):
    benchmark, log_path = run_deterministic_benchmark(model_name, params)
    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

    # Check args
    assert benchmark._args.deterministic is True
    assert getattr(benchmark._args, "generate_log", False) is True
    assert benchmark._args.deterministic is True
    assert benchmark._args.deterministic_seed == 42
    assert benchmark._args.check_frequency == 10

    # Log-file generation and contents
    assert os.path.exists(log_path)
    with open(log_path, "r") as f:
        data = json.load(f)
    assert "schema_version" in data
    assert "metadata" in data
    assert "per_step_fp32_loss" in data
    assert "fingerprints" in data
    assert isinstance(data["per_step_fp32_loss"], list)
    assert isinstance(data["fingerprints"], dict)

    # Run with compare-log for success
    extra_args = f"--compare-log {log_path} --check_frequency 10"
    benchmark_compare, _ = run_deterministic_benchmark(
        model_name, params, log_path, extra_args
    )
    assert benchmark_compare and benchmark_compare.return_code == ReturnCode.SUCCESS

    os.remove(log_path)


@pytest.mark.parametrize("model_name, params", MODELS)
@pytest.mark.xfail(reason="Intentional determinism mismatch to test failure handling.")
def test_pytorch_model_determinism_failure_case(model_name, params):
    benchmark, log_path = run_deterministic_benchmark(model_name, params)
    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

    # Modify the log file to break determinism by changing fingerprints['loss']
    with open(log_path, "r+") as f:
        data = json.load(f)
        # Change the first value in fingerprints['loss']
        if data["fingerprints"]["loss"]:
            data["fingerprints"]["loss"][0] += 1e-5
        else:
            data["fingerprints"]["loss"].append(999.0)
        f.seek(0)
        json.dump(data, f)
        f.truncate()

    # Run with compare-log for failure
    extra_args = f"--compare-log {log_path} --check_frequency 10"
    with pytest.raises(RuntimeError):
        run_deterministic_benchmark(model_name, params, log_path, extra_args)

    # Clean up
    os.remove(log_path)


@pytest.mark.parametrize("model_name, params", MODELS)
def test_pytorch_model_nondeterministoc_default(model_name, params):

    context = BenchmarkRegistry.create_benchmark_context(
        model_name,
        platform=Platform.CUDA,
        parameters=params,
        framework=Framework.PYTORCH,
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (
        benchmark and benchmark.return_code == ReturnCode.SUCCESS
    ), "Benchmark did not run successfully."
    args = benchmark._args
    assert args.deterministic is False, "Expected deterministic to be False by default."
    assert (
        getattr(args, "generate_log", False) is False
    ), "Expected generate_log to be False by default."
    assert (
        getattr(args, "log_path", None) is None
    ), "Expected log_path to be None by default."
    assert (
        getattr(args, "compare_log", None) is None
    ), "Expected compare_log to be None by default."
    assert (
        getattr(args, "check_frequency", None) == 100
    ), "Expected check_frequency to be 100 by default."

    # Periodic fingerprints exist but are empty when not deterministic
    assert hasattr(
        benchmark, "_model_run_periodic"
    ), "Benchmark missing _model_run_periodic attribute."
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict), "_model_run_periodic should be a dict."
    for key in ("loss", "act_mean", "step"):
        assert key in periodic, f"Key '{key}' missing in _model_run_periodic."
        assert (
            len(periodic[key]) == 0
        ), f"Expected empty list for periodic['{key}'], got {periodic[key]}."
    pass
