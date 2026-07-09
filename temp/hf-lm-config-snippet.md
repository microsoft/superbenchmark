# HF GPT/Llama Benchmark — Config Snippet to Copy

The branch does **not** ship a config file (`gb300_config.yaml` is untracked and not pushed).
To run the new HuggingFace GPT/Llama benchmark, copy the pieces below into your own
SuperBench config `.yaml`.

## Prerequisites (one-time, on the machine you run from)

```bash
# 1) Make the sb CLI use this branch's code (adds the new pytorch-hf-lm benchmark)
cd /path/to/superbenchmark
pip install -e . --no-deps --no-build-isolation

# 2) For gated models (meta-llama/*), persist your HF token to disk
huggingface-cli login --token <YOUR_HF_TOKEN>
```

## 1. Enable the benchmark

Add this line under `superbench.enable:` in your config:

```yaml
superbench:
  enable:
    - model-benchmarks:hf-lm-inference
```

## 2. Add the benchmark definition

Add this block under `superbench.benchmarks:`:

```yaml
    # HuggingFace generic LM inference.
    # Pull ANY GPT/Llama checkpoint from the Hub via `model_identifier`.
    # Served by PytorchHFLanguageModel (benchmark name: pytorch-hf-lm).
    # Gated models (meta-llama/*) require the HF token (see prerequisites).
    model-benchmarks:hf-lm-inference:
      modes:
      - name: local
        proc_num: 1
        prefix: CUDA_VISIBLE_DEVICES=0
        parallel: no
      frameworks: [pytorch]
      timeout: 1800
      models:
      - hf-lm
      parameters:
        duration: 0
        num_warmup: 2
        num_steps: 10
        sample_count: 512
        batch_size: 4
        precision: [float16]
        model_action: [inference]
        pin_memory: yes
        num_workers: 0
        seq_len: 128
        model_source: huggingface
        model_identifier: meta-llama/Llama-2-7b-hf   # <-- change to any GPT/Llama model
```

## 3. Run it

```bash
# Local, no orchestration:
sb exec -c <your-config>.yaml --output-dir /tmp/out

# Full orchestration to localhost:
sb run --no-docker -l localhost -c <your-config>.yaml
```

## Swapping models

Just change `model_identifier`. Examples that fit a single GB300 for **inference**:

| model_identifier | Params | Gated? |
|------------------|--------|--------|
| `openai-community/gpt2` | 124M | no |
| `EleutherAI/gpt-j-6b` | 6B | no |
| `EleutherAI/gpt-neox-20b` | 20B | no |
| `meta-llama/Llama-3.2-1B` | 1B | yes |
| `meta-llama/Llama-2-7b-hf` | 7B | yes |
| `meta-llama/Llama-2-13b-hf` | 13B | yes |
| `meta-llama/Llama-2-70b-hf` | 70B | yes (inference-only on 1 GPU) |

To run several models in one entry, list the pre-baked names instead (no `model_identifier`
needed — it's baked into each registration):

```yaml
      models:
      - hf-llama2-7b
      - hf-gpt2
      - hf-gpt-neox-20b
      parameters:
        model_action: [inference]
        precision: [float16]
        batch_size: 4
        seq_len: 128
        num_warmup: 2
        num_steps: 10
        duration: 0
```

## Notes

- `model_action: [inference]` activates the inference-sized memory pre-check, so large
  models (e.g. 70B) that fit for inference but not training are allowed.
- For training instead, set `model_action: [train]` — but expect large models to be
  rejected by the memory gate (70B training ≈ 560GB, won't fit one GPU).
- Precision options for this benchmark: `float32`, `float16`, `bfloat16`.
