# HuggingFace Integration for SuperBench — Feature Context

## Overview

This feature adds the ability to load models directly from [HuggingFace Hub](https://huggingface.co/models) into SuperBench's benchmarking infrastructure, alongside the existing in-house model definitions. It covers three benchmark types:

1. **Model Benchmarks** — Training/inference throughput (BERT, GPT2, Llama, LSTM, CNN)
2. **ORT Inference** — ONNX Runtime inference latency
3. **TensorRT Inference** — TensorRT inference latency

**Branch:** `hf-models-clean`
**Files changed:** 17 (1,769 lines added)

---

## Architecture

### Core Components

```
superbench/benchmarks/micro_benchmarks/
├── model_source_config.py          # Configuration dataclass for model source
├── huggingface_model_loader.py     # HF Hub model loading, caching, validation
└── _export_torch_to_onnx.py        # Extended with export_huggingface_model()

superbench/benchmarks/model_benchmarks/
├── pytorch_base.py                 # Base class: HF args, _create_model_source_config(),
│                                   #   _create_huggingface_model()
├── pytorch_bert.py                 # _create_model_wrapper() for BERT
├── pytorch_gpt2.py                 # _create_model_wrapper() for GPT2
├── pytorch_llama.py                # _create_model_wrapper() for Llama
├── pytorch_lstm.py                 # _create_model_wrapper() for LSTM
└── pytorch_cnn.py                  # _create_model_wrapper() for CNN (ResNet/DenseNet)
```

### Data Flow

```
User passes --model_source huggingface --model_identifier org/model-name
    │
    ▼
PytorchBase.add_parser_arguments()  ← registers --model_source, --model_identifier, --hf_token
    │
    ▼
ModelClass._create_model(precision)
    │
    ├─ if in-house → _create_inhouse_model(precision)     [original path, unchanged]
    │
    └─ if huggingface → PytorchBase._create_huggingface_model(model_config, precision)
           │
           ├─ Creates HuggingFaceModelLoader
           ├─ Calls loader.load_model_from_config(ModelSourceConfig)
           ├─ Calls ModelClass._create_model_wrapper(hf_model, hf_config)
           │      └─ Each model class wraps the HF model with a classification head
           ├─ Sets precision and moves to GPU
           └─ Creates target tensor for training
```

### ONNX Export Flow (ORT / TensorRT)

```
User passes --model_source huggingface --model_identifier org/model-name
    │
    ▼
ORTInferenceBenchmark._preprocess() / TensorRTInferenceBenchmark._preprocess()
    │
    ├─ if in-house → original torchvision export path
    │
    └─ if huggingface → _preprocess_huggingface_models()
           │
           ├─ Loads model on CPU via HuggingFaceModelLoader (avoids device_map issues)
           ├─ Exports via torch2onnxExporter.export_huggingface_model()
           │      ├─ Auto-detects NLP vs vision models (main_input_name)
           │      ├─ Creates appropriate wrapper (NLPModelWrapper / VisionModelWrapper)
           │      ├─ Handles large models (>2GB) with external data format
           │      └─ Disables use_cache to avoid DynamicCache ONNX issues
           ├─ Uses per-rank output dirs to avoid write race conditions
           └─ Optionally applies INT8 quantization
```

---

## Key Classes

### `ModelSourceConfig` (`model_source_config.py`)
Dataclass that encapsulates model loading configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | str | `'in-house'` | `'in-house'` or `'huggingface'` |
| `identifier` | str | required | Model name or HF model ID |
| `hf_token` | str | None | Auth token for gated models |
| `torch_dtype` | str | `'float32'` | `float32`, `float16`, `bfloat16`, `int8` |
| `revision` | str | None | Specific model version/commit |
| `device_map` | str | `'auto'` | Device mapping strategy |
| `cache_dir` | str | None | Model cache directory |

### `HuggingFaceModelLoader` (`huggingface_model_loader.py`)
Handles downloading, caching, and loading models from HuggingFace Hub.

Key methods:
- `load_model(model_identifier, torch_dtype, device, ...)` → `(model, config, tokenizer)`
- `load_model_from_config(ModelSourceConfig)` → `(model, config, tokenizer)`
- `_get_torch_dtype(dtype_str)` → `torch.dtype`
- `_get_model_size(model)` → float (millions of parameters)

Custom exceptions: `ModelLoadError`, `ModelNotFoundError`, `ModelIncompatibleError`

### `PytorchBase` additions (`pytorch_base.py`)
Base class methods added for HF integration:

- `add_parser_arguments()` — adds `--model_source`, `--model_identifier`, `--hf_token`
- `_create_model_source_config(precision)` → `ModelSourceConfig` or `None`
- `_create_huggingface_model(model_config, precision)` → `bool`

Each model subclass implements `_create_model_wrapper(hf_model, hf_config)` to wrap the raw HF model with a task-specific head (e.g., classification linear layer).

---

## CLI Arguments

### Model Benchmarks (pytorch_base.py)

```bash
python3 -m superbench.benchmarks ... \
    --model_source huggingface \
    --model_identifier org/model-name \
    --hf_token $HF_TOKEN               # optional, also reads HF_TOKEN env var
```

### ORT Inference (ort_inference_performance.py)

```bash
# Additional args beyond model_source/model_identifier/hf_token:
    --precision float16|float32|int8 \
    --batch_size 32 \
    --seq_length 512
```

### TensorRT Inference (tensorrt_inference_performance.py)

```bash
# Additional args beyond model_source/model_identifier/hf_token:
    --precision fp16|fp32|int8 \
    --batch_size 32 \
    --seq_length 512 \
    --iterations 2048
```

**Note:** The `--model_source` default is `'in-house'` everywhere, so all existing workflows are unaffected.

---

## Testing

### Unit Tests

| File | What it tests |
|------|---------------|
| `test_model_source_config.py` | Config validation, defaults, deprecated args, is_huggingface() |
| `test_huggingface_loader.py` | Loader init, dtype conversion, mock model loading, error handling |

Run: `pytest tests/benchmarks/micro_benchmarks/test_model_source_config.py tests/benchmarks/micro_benchmarks/test_huggingface_loader.py -v`

### E2E Integration Tests

| File | What it tests |
|------|---------------|
| `test_huggingface_e2e.py` | Actual model downloads from HF Hub (prajjwal1/bert-tiny, distilgpt2) |

These are marked `@pytest.mark.integration` and skipped in CI (require network + disk).

Run: `pytest -m integration tests/benchmarks/micro_benchmarks/test_huggingface_e2e.py -v`

### Manual Testing Checklist

For verifying on a GPU VM:

```bash
# 1. Model benchmark — BERT from HuggingFace
python3 examples/benchmarks/pytorch_huggingface_models.py --model bert

# 2. Model benchmark — GPT2 from HuggingFace
python3 examples/benchmarks/pytorch_huggingface_models.py --model gpt2

# 3. ORT inference — HuggingFace model
python3 examples/benchmarks/ort_inference_performance.py \
    --model_source huggingface --model_identifier bert-base-uncased

# 4. TensorRT inference — HuggingFace model
python3 examples/benchmarks/tensorrt_inference_performance.py \
    --model_source huggingface --model_identifier bert-base-uncased

# 5. ORT inference — in-house model (verify no regression)
python3 examples/benchmarks/ort_inference_performance.py

# 6. TensorRT inference — in-house model (verify no regression)
python3 examples/benchmarks/tensorrt_inference_performance.py

# 7. Vision model (CNN) via ORT
python3 examples/benchmarks/ort_inference_performance.py \
    --model_source huggingface --model_identifier microsoft/resnet-50

# 8. Gated model (requires HF_TOKEN)
HF_TOKEN=hf_xxx python3 examples/benchmarks/pytorch_huggingface_models.py --model bert
```

---

## Important Design Decisions

1. **CPU loading for ONNX export**: HF models are loaded on CPU before ONNX export to avoid `accelerate` dispatching across GPUs, which causes device mismatch errors during export.

2. **Per-rank output dirs**: In distributed environments, each process writes ONNX files to `rank_{N}/` subdirectories to avoid write race conditions.

3. **Model wrappers**: Each model class defines its own `_create_model_wrapper()` because different architectures use different output formats (pooler output for BERT, last hidden state for GPT2, etc.).

4. **NLP vs Vision auto-detection**: ONNX export uses `model.main_input_name` to detect whether a model expects `input_ids` (NLP) or `pixel_values` (vision), and creates appropriate dummy inputs and wrappers.

5. **Large model support**: Models >2GB are exported with ONNX external data format automatically.

6. **Backward compatibility**: Default `--model_source` is `'in-house'` everywhere, so all existing benchmarks work unchanged.

---

## Known Limitations

- Not all HF model architectures are guaranteed to work with ONNX export (some use dynamic control flow)
- The `huggingface_model_loader.py` always logs an ONNX compatibility warning even when ONNX export isn't being used
- `model_source_config.py:65` uses `tuple[bool, str]` type hint which requires Python 3.9+
- The `__init__.py` in `model_benchmarks/` has a pre-existing issue where `PytorchMixtral` is appended to `__all__` twice

---

## Automated Validation Script

A self-contained test script `test_hf_integration.py` is provided at the repo root. It validates the integration in three layers:

| Layer | What it tests | Requirements |
|-------|--------------|--------------|
| 1 | Unit tests — config validation, loader init, dtype conversion, imports | None |
| 2 | Integration — download models from HF Hub, forward pass, error handling | Network |
| 3 | GPU benchmarks — model/ORT/TensorRT benchmarks with HF + in-house regression | GPU + Network |

```bash
# Run all layers (auto-detects resources):
python3 test_hf_integration.py

# Unit tests only (no GPU/network):
python3 test_hf_integration.py --layer 1

# Unit + integration (network, no GPU):
python3 test_hf_integration.py --layer 2

# Everything including GPU benchmarks:
python3 test_hf_integration.py --layer 3
```

The script prints `[PASS]`/`[FAIL]`/`[SKIP]` per test and a summary at the end. Exit code is 0 if all passed, 1 if any failed.

---

## Adding a New Model

To add HuggingFace support for a new model benchmark (e.g., `pytorch_newmodel.py`):

1. Add `_create_model_wrapper(self, hf_model, hf_config)` method that wraps the HF model with your task-specific head
2. Update `_create_model(self, precision)` to check for HF source:
   ```python
   def _create_model(self, precision):
       model_config = self._create_model_source_config()
       if model_config and model_config.is_huggingface():
           return self._create_huggingface_model(model_config, precision)
       return self._create_inhouse_model(precision)
   ```
3. Move original model creation logic into `_create_inhouse_model(self, precision)`
4. The base class (`PytorchBase`) handles everything else: argument parsing, HF loading, precision, GPU placement, target tensor creation

---

## Pre-Download GPU Memory Check (OOM Prevention)

### Problem
Large HuggingFace models (e.g. Llama-2-70B, GPT-NeoX-20B) take a long time to download. If the model won't fit on the available GPU, the user wastes time downloading tens of GB of weights only to hit an OOM error during model instantiation or training.

### Solution
A two-step pre-flight check that estimates whether the model will fit **before** downloading weights. The estimation logic lives in shared static methods on `HuggingFaceModelLoader` and is used by all three benchmark types (PyTorch training, ORT inference, TensorRT inference).

1. **Config-only download** — `AutoConfig.from_pretrained()` downloads just the model config (a few KB), not the weights.
2. **Parameter count estimation** — `HuggingFaceModelLoader.estimate_param_count_from_config(hf_config)` computes an architecture-aware estimate covering embeddings, transformer layers (attention + MLP), layer norms, LM head, and MoE experts.
3. **Memory estimation** — `HuggingFaceModelLoader.estimate_memory(param_count, precision, mode)` multiplies by a mode-dependent factor (4x for training, 1.2x for inference) and compares against 85% of available GPU memory (or system RAM if no GPU).
4. **Graceful failure** — If the model won't fit, logs a detailed error message with actionable suggestions and returns `False` without downloading any weights.

### Shared Methods in `HuggingFaceModelLoader` (`huggingface_model_loader.py`)

| Method | Purpose |
|--------|---------|
| `estimate_param_count_from_config(hf_config)` | Static. Estimates parameter count from HF config attributes (vocab_size, hidden_size, num_hidden_layers, etc.) without instantiating the model. Handles GQA (grouped-query attention), gated MLPs (SiLU/SwiGLU), and MoE architectures. Returns 0 if estimation fails. |
| `estimate_memory(param_count, precision_str, mode)` | Static. Computes estimated memory — 4x model size for training (weights + gradients + Adam), 1.2x for inference (weights + overhead). Uses `torch.cuda.get_device_properties(0).total_memory` on GPU, or `min(system_ram, 80GB)` on CPU. Returns `(estimated_bytes, available_bytes, fits_bool)`. |
| `check_memory_fits(model_identifier, hf_config, precision_str, mode, token)` | Static convenience wrapper. Calls the above two methods, logs the result, and returns `(fits, param_millions, estimated_gb, available_gb)`. |

### Callers

| Benchmark | File | Where | Mode |
|-----------|------|-------|------|
| PyTorch training | `pytorch_base.py` | `_create_huggingface_model()` | `'training'` |
| Mixtral MoE | `pytorch_mixtral_impl.py` | `_preprocess()` | `'training'` (via inherited delegate methods) |
| ORT inference | `ort_inference_performance.py` | `_preprocess_huggingface_models()` | `'inference'` |
| TensorRT inference | `tensorrt_inference_performance.py` | `_preprocess_huggingface_models()` | `'inference'` |

### Updated Flow (PyTorch Training Example)

```
Step 1: AutoConfig.from_pretrained() — download config only (few KB)
    │
Step 2: HuggingFaceModelLoader.estimate_param_count_from_config(hf_config)
    │
Step 3: HuggingFaceModelLoader.estimate_memory(param_count, precision, 'training')
    │
    ├─ if NOT fits → log error with suggestions, return False (no weight download)
    │
    └─ if fits → proceed with HuggingFaceModelLoader.load_model_from_config()
         │
         Step 4: _create_model_wrapper() → set precision → .train() → .cuda()
```

### Testing

The `model-benchmarks:gpt-neox-hf` config entry in `gb200_config.yaml` tests this with `EleutherAI/gpt-neox-20b` (20B params, ~329GB training memory in float32). This triggers the pre-download rejection on any single GPU.

Validated results on GB200 (~197.9GB VRAM):
- GPT-NeoX-20B (float32, training): 20,551.6M params, ~328.8GB estimated → rejected
- Mixtral-8x7B (float16, training): 46,702.8M params, ~373.6GB estimated → rejected
