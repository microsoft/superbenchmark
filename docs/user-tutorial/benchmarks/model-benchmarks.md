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

## Official-checkpoint V2-Lite Training

### `deepseek-v2-lite-training`

This opt-in, standalone MODEL benchmark downloads and hash-verifies the complete
`deepseek-ai/DeepSeek-V2-Lite` checkpoint at
`604d5664dddd88a0433dbae533b7fe9472482de0`, then performs full-parameter causal-LM
training. It is not the Megatron benchmark, a classifier-head workload, LoRA, or
the reduced hidden-size configuration from PR #765.

The model has 15,706,484,224 parameters, hidden size 2048, 27 layers, MLA and MoE.
The four BF16 weight files total 31,413,626,576 bytes. Initial optimizer state is
new: this is short training from pretrained weights, not a resume of DeepSeek's
original optimizer or an attempt to train a converged model within an hour.
The small `fixed-text-v1` fixture uses the official tokenizer and packed token
sequences without padding. It is reproducible, not a production-data accuracy
or representativeness evaluation. Mock data and mocked subprocesses are used
only in auxiliary unit tests.

#### Validation Status

This is a draft implementation with historical four-H100 full-model worker and
real frontend evidence, not complete qualification. The current global-batch
reference changes, repeated cold runs, steady-state utilization and final
reproducible controller runtime still require hardware acceptance. This is not
cross-SKU qualification.

#### Runtime And Execution

Use a separate worker environment; do not replace the legacy SuperBench
framework dependencies. The worker requires Python 3.10+, a GPU-compatible
PyTorch 2.5+ build, and the pinned packages in
[`deepseek-v2-lite-reference.txt`](../../../dockerfile/requirements/deepseek-v2-lite-reference.txt).
The supplied
[`model runtime Dockerfile`](../../../dockerfile/deepseek-v2-lite-training.dockerfile)
pins the CUDA PyTorch base by digest and installs the Python SuperBench
controller, runner and diagnosis entrypoints together with the worker/reference
environment. The
[`controller lock`](../../../dockerfile/requirements/deepseek-v2-lite-controller.txt)
is specific to Python 3.12/Linux x86_64 and retains the base image's GPU
Torch/torchvision builds. Its regeneration command is in the
[`lock inputs`](../../../dockerfile/requirements/deepseek-v2-lite-controller.in).
This opt-in model runtime does not replace the standard images or provide the
complete compiled microbenchmark suite.
The build checks the Torch/NumPy array bridge and the real runner/diagnosis CLI,
not just successful package installation.

```bash
docker build -f dockerfile/deepseek-v2-lite-training.dockerfile -t sb-v2lite-training .
docker run --rm --gpus '"device=0,1,2,3"' --shm-size 32g \
  -v "$PWD/v2lite-results:/results" sb-v2lite-training exec \
  --config-file superbench/config/deepseek-v2-lite-training.yaml \
  --output-dir /results/sb-output \
  -C superbench.benchmarks.deepseek-v2-lite-training.parameters.cache_root=/results/model-cache
```

For a separately prepared native installation:
```bash
python3 -m venv --system-site-packages /opt/deepseek-reference
/opt/deepseek-reference/bin/pip install -r dockerfile/requirements/deepseek-v2-lite-reference.txt
sb exec --config-file superbench/config/deepseek-v2-lite-training.yaml
```

The native command assumes SuperBench is already installed in the controller's
environment. Change `runtime_python` if the prepared worker environment lives
elsewhere. Install a compatible CUDA/ROCm PyTorch build before preparing the
worker, rather than letting a generic package install choose the GPU backend.

Select the
[`training profile`](../../../superbench/config/deepseek-v2-lite-training.yaml)
as its own benchmark entry, with one local launcher. The worker launches
`torchrun --standalone` across `num_gpus` local GPUs. Do not launch this entry
once per GPU or combine it with other models, precisions or actions under the
same deadline. `batch_size` is the microbatch per GPU.

FSDP shards the full model, gradients and FP32 AdamW master/optimizer state.
Activation checkpointing is enabled for the measured workload. This is not
expert parallelism and does not qualify a Megatron or all-to-all dispatch path.
The default example uses four GPUs and microbatch two, retaining a global batch
of eight sequences. The complete profile also runs an unsharded BF16 loss and
gradient reference on one GPU; use 80GB-class GPUs for this initial example.
This is not a minimum-card proof or a recommendation to replicate the full
model with DDP: FP32 AdamW training state must fit across the selected shards.
Registration for CUDA/ROCm denotes an implementation route, not certification
of every GPU, runtime or memory configuration.

#### Correctness, Timing And Results

Before optimization, an unsharded official-code BF16 reference checks the
complete model's loss and sampled embedding, dense, MLA, router and active-expert
gradients against FSDP. Gradient probes include nonzero and distributed positions;
they are not an exhaustive comparison of every parameter. Every training step
requires finite loss and a finite nonzero global gradient norm. The first update
must change model parameters. No checkpoint missing-key or shape mismatch is
accepted.
The reference preserves every sequence of the actual global batch and its full
context length, evaluating at most two sequences at a time on one GPU with
activation checkpointing. Sequence-count-weighted loss and gradient probes are
accumulated in FP32, including partial final chunks. The pinned zero-dropout,
per-sequence auxiliary-loss configuration makes this decomposition valid.
Per-rank and global losses are checked separately. Replicating one identical
short input across ranks would not establish this distributed-reduction contract.
The one-time pre-update parameter snapshot is stored in host memory and checked
after the update. Model parameters, gradients and optimizer state remain on the
GPUs; this is validation storage, not training-state CPU offload.

Every invocation creates its own empty HF/Hub/Xet/module cache and keeps its
report and logs under a new directory in `cache_root`. Shared caches are not
deleted. `duration` covers download, SHA256 verification, loading, correctness,
warmup, measured training and worker/rank teardown. It must be below 3600 seconds;
the controller reserves 20 seconds for cleanup. A watchdog exit is a failure.
Image pulling, dependency preparation and SuperBench installation are separate
runtime-preparation steps, not certified by the model-cold timer.

A dedicated Linux child supervisor reaps the complete worker tree, including
torchrun ranks that start separate sessions. A launcher exit with surviving
descendants is a failure; timeout cleanup escalates from termination to forced
exit and verifies that owned children were reaped without targeting other jobs.
The worker records Unix timestamps bounding the measured phase for external
GPU telemetry; the step durations continue to use the monotonic clock.

| Metric | Unit | Scope |
| --- | --- | --- |
| `bf16_train_step_time` | ms | Mean of the slowest synchronized rank per measured step |
| `bf16_train_tokens_per_second` | loss tokens/s | Global next-token targets, excluding each sequence's first position |
| `bf16_train_loss` | loss | Global mean causal-LM loss |
| `bf16_train_gradient_norm` | norm | Global gradient norm before clipping |
| `end_to_end_seconds` | s | Complete worker launch through exit and report validation |
| `checkpoint_bytes` | bytes | Complete weight files, not measured network traffic or GPU memory |

Step timing includes forward, backward, clipping and the optimizer update.
Input preparation/copy and reference/update checks are excluded from the
steady step metric but included in the complete-job time. Keep model-cold
acceptance, per-GPU utilization and model quality as separate conclusions.
