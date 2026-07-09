# HuggingFace GPT/Llama Model Benchmarks — Handoff Context

> Agent context doc for continuing work on the `atonpe/hf-pytorch-models` branch.
> Companion to `temp/hf-int-context.md` (the original HF-integration context).
> This doc covers the **generic GPT/Llama HuggingFace wrapper** added on top of that base.

---

## TL;DR — what this change adds

A single generic PyTorch model benchmark, `PytorchHFLanguageModel`, that pulls **any
GPT- or Llama-style checkpoint** from the HuggingFace Hub and benchmarks its
train/inference throughput exactly like the other SuperBench PyTorch model benchmarks.
A model only runs if a **pre-flight memory check** estimates it fits on the available
hardware (mode-aware: inference vs training).

It also **fixes a broken branch state**: the previous commit imported a
`pytorch_deepseek` module that was never committed, so `superbench.benchmarks.model_benchmarks`
failed to import at all (`ModuleNotFoundError`). That dangling import is removed.

---

## Files changed

| File | Type | What |
|------|------|------|
| `superbench/benchmarks/model_benchmarks/pytorch_hf_lm.py` | **NEW** | `PytorchHFLanguageModel` class + all `pytorch-hf-*` registrations |
| `superbench/benchmarks/model_benchmarks/__init__.py` | modified | Removed dangling `pytorch_deepseek` import; fixed duplicate `PytorchMixtral` in `__all__`; export `PytorchHFLanguageModel` |
| `superbench/benchmarks/model_benchmarks/pytorch_base.py` | modified | Inference-aware memory gate (see below) |
| `examples/benchmarks/pytorch_huggingface_models.py` | modified | Added GPT/Llama entries + `HF_TOKEN` docs |
| `tests/benchmarks/model_benchmarks/test_pytorch_determinism_all.py` | modified | Removed the never-registered `deepseek-r1-distill-qwen-1.5b` case |

> NOTE: `gb300_config.yaml` is **untracked** and intentionally **not** pushed. The config
> entry used to drive these runs is provided separately in `temp/hf-lm-config-snippet.md`.

---

## The new class — `PytorchHFLanguageModel`

File: `superbench/benchmarks/model_benchmarks/pytorch_hf_lm.py`

- Subclasses `PytorchBase`; HuggingFace-only (`_create_model()` errors if
  `--model_source huggingface` + `--model_identifier` aren't provided).
- `_create_model_wrapper(hf_model, hf_config)` — **architecture-agnostic** classification-head
  wrapper. Works for both GPT and Llama because both return the last hidden state as
  `outputs[0]`. Hidden size is detected via `hidden_size` → `n_embd` → `d_model`.
- Supported precisions: `FLOAT32`, `FLOAT16`, `BFLOAT16` (no FP8 on the generic HF path).
- Standard `_train_step` / `_inference_step` loops (copied from the Llama/GPT2 benchmarks).
- Random token dataset via `TorchRandomDataset` (`--seq_len`, `--num_classes`).

### Registered benchmark names

Generic (supply the model ID yourself via `--model_identifier`):
- `pytorch-hf-lm`  ← `--model_source huggingface` baked in; identifier comes from params/config

Pre-baked (identifier already set — just name it and run):
- GPT (public): `pytorch-hf-gpt2`, `-gpt2-medium`, `-gpt2-large`, `-gpt2-xl`, `-gptj-6b`, `-gpt-neox-20b`
- Llama (gated, need `HF_TOKEN`): `pytorch-hf-llama2-7b`, `-llama2-13b`, `-llama2-70b`,
  `-llama3-8b`, `-llama3.1-8b`, `-llama3.2-1b`, `-llama3.2-3b`

---

## The inference-aware memory gate (in `pytorch_base.py`)

The base pre-flight OOM check estimates memory from the model config (small download)
BEFORE pulling weights, and rejects models that won't fit. It now picks the mode from
`--model_action`:

- New helper `_get_hf_memory_mode()` → returns `'inference'` when only inference is
  requested, else `'training'` (default).
- `_estimate_training_memory` was renamed to `_estimate_hf_memory(param_count, precision, mode=...)`.
- `_create_huggingface_model()` uses the mode; log/error messages say "training"/"inference"
  accordingly.

Why it matters: training needs ~4× params (weights + grads + 2× Adam moments); inference
needs ~1.2× (weights + overhead). A 70B model is ~560GB for training (rejected on one GPU)
but ~168GB for inference (fits on a GB300). Without the mode switch, 70B inference would be
wrongly rejected. Default behavior for existing benchmarks (training) is unchanged.

Memory math lives in `HuggingFaceModelLoader.estimate_memory()`
(`superbench/benchmarks/micro_benchmarks/huggingface_model_loader.py`):
`bytes = params × bytes_per_param × multiplier`, fits if `bytes / gpu_total < 0.85`.
`bytes_per_param`: fp16/bf16=2, fp32=4, int8=1. `multiplier`: training=4, inference=1.2.

---

## How to run

### A) Programmatic (registry API) — fastest for a quick check
```python
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework
ctx = BenchmarkRegistry.create_benchmark_context(
    'hf-llama2-7b', platform=Platform.CUDA,
    parameters='--batch_size 4 --seq_len 128 --num_warmup 2 --num_steps 10 '
               '--model_action inference --precision float16 --duration 0',
    framework=Framework.PYTORCH)
b = BenchmarkRegistry.launch_benchmark(ctx)
print(b.return_code, b.result)
```

### B) Config-driven, local, no orchestration
```bash
sb exec -c <config>.yaml --output-dir /tmp/out
```

### C) Full orchestration (ansible → localhost) — the production path
```bash
sb run --no-docker -l localhost -c <config>.yaml
```

Config `models: [hf-lm]` → executor prepends framework → `pytorch-hf-lm`. The `parameters:`
dict is converted to CLI flags, so `model_identifier: meta-llama/Llama-2-7b-hf` becomes
`--model_identifier meta-llama/Llama-2-7b-hf`.

---

## IMPORTANT environment gotchas (needed for `sb exec` / `sb run`)

1. **Editable install** — the `sb` CLI uses the *installed* `superbench` package, not the
   workspace source. After adding a new benchmark file you MUST make the install point at
   the workspace, or `sb` won't see `pytorch-hf-lm`:
   ```bash
   pip install -e . --no-deps --no-build-isolation
   ```
   (Symptom if you skip this: `registry.py: Benchmark has no implementation, name: pytorch-hf-lm, platform: CUDA`.)

2. **Gated models (Llama)** need a token. `sb run` spawns a fresh process via ansible, so
   env vars may not propagate — persist the token to disk:
   ```bash
   huggingface-cli login --token <HF_TOKEN>   # writes ~/.cache/huggingface/token
   ```
   The loader reads `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` from env, and `huggingface_hub`
   falls back to the on-disk token automatically.

---

## What was validated (on this 4× GB300 node, ~297GB/GPU)

All `return_code: 0`, fp16 inference, `--model_action inference`:

| Model | Params | Path used | fp16 step time | Throughput |
|-------|--------|-----------|----------------|------------|
| meta-llama/Llama-3.2-1B | 1.2B | registry + `sb exec` + `sb run` | 13.3 ms | ~302 /s |
| meta-llama/Llama-2-7b-hf | 6.6B | `sb run` (fresh download) | 25.2 ms | ~159 /s |
| meta-llama/Llama-2-70b-hf | 68.7B | registry API | 70.9 ms | ~56 /s |

Static checks: package imports clean, all `pytorch-hf-*` contexts resolve, no lint errors,
no stray `deepseek` refs in pytorch model code.

**Not yet done:** a dedicated unit test for `PytorchHFLanguageModel`; GPT-family end-to-end
run; 70B via the config path (only via registry API so far).

---

## GB300 fit guidance (single GPU, DDP replicates — no model parallelism here)

- **Training** fits up to ~GPT-NeoX-20B (fp16) / Llama-2-13B. 70B training does NOT fit.
- **Inference** fits everything listed, including 70B (fp16 ~168GB, int8 ~84GB).

---

## Adding another pre-baked model

Append one registration in `pytorch_hf_lm.py`:
```python
BenchmarkRegistry.register_benchmark(
    'pytorch-hf-<name>',
    PytorchHFLanguageModel,
    parameters='--model_source huggingface --model_identifier <org/model-name>',
)
```
No new class needed — the generic wrapper handles any GPT/Llama decoder LM.

---

## Suggested cleanup before pushing

- Confirm `gb300_config.yaml`, `docker-build.log`, and `temp/` stay untracked (don't push them).
- `git add superbench/benchmarks/model_benchmarks/pytorch_hf_lm.py` (new file — currently untracked).
- Consider adding a unit test for `PytorchHFLanguageModel`.
