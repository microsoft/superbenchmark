---
id: model-benchmarks
---

# Model Benchmarks

## PyTorch Model Benchmarks

### `model-benchmarks`

#### Introduction

Run training or inference tasks with single or half precision for deep learning models,
including the following categories:
* GPT: gpt2-small, gpt2-medium, gpt2-large and gpt2-xl
* LLAMA: llama2-7b, llama2-13b, llama2-70b
* MoE: mixtral-8x7b, mixtral-8x22b
* BERT: bert-base and bert-large
* LSTM
* CNN, listed in [`torchvision.models`](https://pytorch.org/vision/0.8/models.html), including:
  * resnet: resnet18, resnet34, resnet50, resnet101, resnet152
  * resnext: resnext50_32x4d, resnext101_32x8d
  * wide_resnet: wide_resnet50_2, wide_resnet101_2
  * densenet: densenet121, densenet169, densenet201, densenet161
  * vgg: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19
  * mnasnet: mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
  * mobilenet: mobilenet_v2
  * shufflenet: shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
  * squeezenet: squeezenet1_0, squeezenet1_1
  * others: alexnet, googlenet, inception_v3

For inference, supported percentiles include
50<sup>th</sup>, 90<sup>th</sup>, 95<sup>th</sup>, 99<sup>th</sup>, and 99.9<sup>th</sup>.

**New: Support fp8_hybrid and fp8_e4m3 precision for BERT models.**

**New: Deterministic Training Support**
SuperBench now supports deterministic training to ensure reproducibility across runs. This includes fixed seeds and deterministic algorithms. To enable deterministic training, use the following flags:

- **Flags:**
  - `--enable_determinism`: Enables deterministic computation for reproducible results.
  - `--deterministic_seed <seed>`: Sets the seed for reproducibility (default: 42).
  - `--check_frequency <steps>`: How often to record deterministic metrics (default: 100).

- **Environment Variables (set automatically by SuperBench when `--enable_determinism` is used):**
  - `CUBLAS_WORKSPACE_CONFIG=:4096:8`: Ensures deterministic behavior in cuBLAS. This can be overridden by setting it manually before running SuperBench.

**Comparing Deterministic Results**

To compare deterministic results between runs, use the standard result analysis workflow:

1. Run benchmark with `--enable_determinism` flag
2. Generate baseline: `sb result generate-baseline --data-file results.jsonl --summary-rule-file rules.yaml`
3. Compare future runs: `sb result diagnosis --data-file new-results.jsonl --rule-file diagnosis-rule.yaml --baseline-file baseline.json`

This allows configurable tolerance for floating-point differences via YAML rules.

**Configuration Parameter Validation**

When determinism is enabled, benchmark configuration parameters (batch_size, num_steps, deterministic_seed, etc.) are automatically recorded in the results file as `deterministic_config_*` metrics. The diagnosis rules enforce exact matching of these parameters between runs to ensure valid comparisons:

If any configuration parameter differs between runs, the diagnosis will flag it as a failure, ensuring you only compare runs with identical configurations.

**Summary Rule Snippet for Determinism**

Include the following rule in your summary rule file (used with `sb result summary` or `sb result generate-baseline --summary-rule-file`) to surface deterministic metrics in the results summary:

```yaml
superbench:
  rules:
    model-benchmarks-deterministic:
      statistics:
        - mean
      categories: Deterministic
      metrics:
        - model-benchmarks:.*/deterministic_loss.*
        - model-benchmarks:.*/deterministic_act_mean.*
        - model-benchmarks:.*/deterministic_check_count.*
        - model-benchmarks:.*/deterministic_step.*
        - model-benchmarks:.*/deterministic_config_.*
        - model-benchmarks:.*/return_code.*
```

This groups all deterministic outputs — loss fingerprints, activation means, check counts, step numbers, configuration parameters, and return codes — under the **Deterministic** category.

**Diagnosis Rule Snippet for Determinism**

Include the following rules in your diagnosis rule file (used with `sb result diagnosis` or `sb result generate-baseline --diagnosis-rule-file`) to detect Silent Data Corruption (SDC) and validate configuration consistency:

```yaml
superbench:
  rules:
    deterministic_rule:
      function: variance
      criteria: "lambda x: x != 0"
      categories: SDC-Fingerprint
      metrics:
        - model-benchmarks:.*/deterministic_loss.*
        - model-benchmarks:.*/deterministic_act_mean.*
        - model-benchmarks:.*/deterministic_check_count.*

    deterministic_config_rule:
      function: variance
      criteria: "lambda x: x != 0"
      categories: SDC-Config
      metrics:
        - model-benchmarks:.*/deterministic_config_batch_size.*
        - model-benchmarks:.*/deterministic_config_num_steps.*
        - model-benchmarks:.*/deterministic_config_num_warmup.*
        - model-benchmarks:.*/deterministic_config_deterministic_seed.*
        - model-benchmarks:.*/deterministic_config_check_frequency.*
        - model-benchmarks:.*/deterministic_config_seq_len.*
        - model-benchmarks:.*/deterministic_config_hidden_size.*
        - model-benchmarks:.*/deterministic_config_num_classes.*
        - model-benchmarks:.*/deterministic_config_input_size.*
        - model-benchmarks:.*/deterministic_config_num_layers.*
        - model-benchmarks:.*/deterministic_config_num_hidden_layers.*
        - model-benchmarks:.*/deterministic_config_num_attention_heads.*
        - model-benchmarks:.*/deterministic_config_intermediate_size.*

    deterministic_failure_rule:
      function: failure_check
      criteria: "lambda x: x != 0"
      categories: SDC-Failed
      metrics:
        - model-benchmarks:.*/return_code
```

- **SDC-Fingerprint** (`deterministic_rule`): Flags any node where loss, activation mean, or check count has *any* variance from baseline (`x != 0`), indicating a potential SDC issue.
- **SDC-Config** (`deterministic_config_rule`): Ensures all determinism configuration parameters (seed, batch size, sequence length, hidden size, etc.) are identical across nodes — any mismatch means the comparison is invalid.
- **SDC-Failed** (`deterministic_failure_rule`): Uses `failure_check` to catch nodes where the determinism benchmark failed to run or returned a non-zero exit code.

For complete rule files covering all benchmark categories (micro-benchmarks, NCCL, GPU copy bandwidth, NVBandwidth, etc.), refer to the rule file documentation in [Result Summary](../result-summary.md) and [Data Diagnosis](../data-diagnosis.md).

#### Metrics

| Name                                                                                    | Unit                   | Description                                                                  |
|-----------------------------------------------------------------------------------------|------------------------|------------------------------------------------------------------------------|
| model-benchmarks/pytorch-${model_name}/${precision}_train_step_time                     | time (ms)              | The average training step time with fp32/fp16 precision.                     |
| model-benchmarks/pytorch-${model_name}/${precision}_train_throughput                    | throughput (samples/s) | The average training throughput with fp32/fp16 precision per GPU.            |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_step_time                 | time (ms)              | The average inference step time with fp32/fp16 precision.                    |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_throughput                | throughput (samples/s) | The average inference throughput with fp32/fp16 precision.                   |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_step_time\_${percentile}  | time (ms)              | The n<sup>th</sup> percentile inference step time with fp32/fp16 precision.  |
| model-benchmarks/pytorch-${model_name}/${precision}_inference_throughput\_${percentile} | throughput (samples/s) | The n<sup>th</sup> percentile inference throughput with fp32/fp16 precision. |


## Megatron Model benchmarks

### `megatron-gpt`

#### Introduction

Run GPT pretrain tasks with float32, float16, bfloat16 precisions with [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed).

`tips: batch_size in this benchmark represents global batch size, the batch size on each GPU instance is micro_batch_size.`

#### Metrics
| Name                                              | Unit                   | Description                                             |
|---------------------------------------------------|------------------------|---------------------------------------------------------|
| megatron-gpt/${precision}_train_step_time         | time (ms)              | The average training step time per iteration.           |
| megatron-gpt/${precision}_train_throughput        | throughput (samples/s) | The average training throughput per iteration.          |
| megatron-gpt/${precision}_train_tflops            | tflops/s               | The average training tflops per second per iteration.   |
| megatron-gpt/${precision}_train_mem_allocated     | GB                     | The average GPU memory allocated per iteration.         |
| megatron-gpt/${precision}_train_max_mem_allocated | GB                     | The average maximum GPU memory allocated per iteration. |

## DeepSeek-V2-Lite Training

### `deepseek-v2-lite-training`

#### Introduction

Run full-parameter BF16 causal-LM training with the official
[DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite) checkpoint at revision
`604d5664dddd88a0433dbae533b7fe9472482de0` (15.7B parameters). The worker starts `torchrun` on
`num_gpus` local GPUs (at least two) and trains on a small fixed-text dataset with fully sharded FSDP,
AdamW and activation checkpointing; `batch_size` is the microbatch per GPU. The job downloads the
checkpoint and verifies the SHA256 of its 31.4GB of weight files. Before training, the FSDP loss and
sampled gradients must match an unsharded official-code reference, and every step must produce a
finite loss and a finite, nonzero gradient norm. Training starts from the pretrained weights with a
new optimizer state; it measures training performance, not convergence or model quality.

`moe_dispatch` defaults to `official`. `stable_sorted` groups routed tokens with a stable sort instead
of repeated Boolean indexing; it keeps the same experts, routing and weights, and the reference check
compares it with the official implementation.

Validation covers one profile on four H100 80GB GPUs: `batch_size=36`, `seq_len=512`,
`num_warmup=2`, `num_steps=128`, `moe_dispatch=stable_sorted` and
`PYTORCH_CUDA_ALLOC_CONF=backend:native`. The supplied profile is a smaller example
(`batch_size=2`, `num_steps=8`, official dispatch); other configurations are not validated. The
unsharded reference runs on one GPU, so use 80GB-class GPUs.

#### Usage

Run the [training profile](../../../superbench/config/deepseek-v2-lite-training.yaml) as its own
benchmark entry with one local process (`proc_num: 1`). The
[model runtime Dockerfile](../../../dockerfile/deepseek-v2-lite-training.dockerfile) provides the
worker environment:

```bash
docker build -f dockerfile/deepseek-v2-lite-training.dockerfile -t sb-v2lite-training .
docker run --rm --gpus '"device=0,1,2,3"' --shm-size 32g \
  -v "$PWD/v2lite-results:/results" sb-v2lite-training exec \
  --config-file superbench/config/deepseek-v2-lite-training.yaml \
  --output-dir /results/sb-output \
  -C superbench.benchmarks.deepseek-v2-lite-training.parameters.cache_root=/results/model-cache
```

Outside Docker, set `runtime_python` to a separate Python 3.10+ environment with a GPU build of
PyTorch 2.5+ and the [reference packages](../../../dockerfile/requirements/deepseek-v2-lite-reference.txt).

`cache_root` is required. Each run downloads the checkpoint into a new directory under it and deletes
that model cache at the end, keeping the report and logs. If deletion fails, the cache path is logged
for manual removal. `duration` must be below 3600 seconds and covers the whole job, from download to
cache deletion.

#### Metrics

| Name | Unit | Description |
|---|---|---|
| deepseek-v2-lite-training/bf16_train_step_time | time (ms) | The average training step time of the slowest GPU, including forward, backward, gradient clipping and the optimizer update. |
| deepseek-v2-lite-training/bf16_train_tokens_per_second | throughput (tokens/s) | Next-token targets trained per second across all GPUs. |
| deepseek-v2-lite-training/bf16_train_loss | loss | The average global causal-LM loss. |
| deepseek-v2-lite-training/bf16_train_gradient_norm | norm | The average global gradient norm before clipping. |
| deepseek-v2-lite-training/end_to_end_seconds | time (s) | The complete job time, including checkpoint download and cache deletion. |
| deepseek-v2-lite-training/checkpoint_bytes | size (bytes) | The total size of the checkpoint weight files. |
