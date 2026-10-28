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

## Official-checkpoint V2-Lite Generative Inference

### `deepseek-v2-lite-inference`

This opt-in standalone MODEL benchmark loads the complete official
`deepseek-ai/DeepSeek-V2-Lite` checkpoint at
`604d5664dddd88a0433dbae533b7fe9472482de0`, with four BF16 weight files totaling
31,413,626,576 bytes. It benchmarks prefill and multi-token cached generation,
not repeated forward passes on randomly initialized weights. No Megatron,
training implementation, quantization or CPU offload is required.

Use the
[`inference profile`](../../../superbench/config/deepseek-v2-lite-inference.yaml)
as its own entry, with one local controller and one GPU inference replica.
The supplied profile explicitly selects the qualified H100 numerical policy;
it is not a generic CUDA/ROCm acceptance profile.
`batch_size` is the number of simultaneous requests in each measured batch.
The versioned, small fixed-text fixture controls prompt token lengths; it is
not a model-quality or production-request-distribution benchmark.
Generated counts come from actual output token IDs, including EOS when emitted,
not `batch_size * max_new_tokens`. EOS and maximum-length termination remain
enabled; all-first-token-EOS output does not satisfy the cached-decode workload.

#### Validation Status

This remains a draft implementation. The explicit `h100_bf16_v6` numerical
method has passed independent-reference checks and all six preregistered H100
engine windows: 3,216 positive positions pass and all 6,438 fixed fault cases
are rejected. The recorded public runtime completed a download-inclusive cold
bootstrap in 1323.74 seconds. Its opt-in internal package then passed three
ordinary-entrypoint cold repetitions in 1339.17, 1342.39 and 1329.00 seconds,
with complete diagnosis, 97.85-98.32% mean GPU busy and a maximum throughput
deviation of 0.334% from the median. Each repetition includes all numerical checks.
These results qualify the recorded local source/image/profile snapshot, not an
unchanged older PR head, a later rebuilt image, another SKU or production deployment.
Do not replace numerical validation with matching generated tokens or exit code zero.

The adapter default remains `numerical_policy=strict_fp32`, preserving the
original pointwise FP32 checks. `h100_bf16_v6` is explicitly restricted to one
H100, batch 8, sequence 512, maximum decode 64, two warmup batches and 256
measured batches. It uses compiled, batch-invariant BF16 inference with the
compressed-MLA implementation; there is no FP32 measured-model fallback.

The [shipped numerical contract](../../../superbench/config/deepseek-v2-lite-h100-numerics.json)
is bound to an exact SHA256 by the worker. After engine shutdown, independent
official BF16 cached execution measures the precision discrepancy on the same
fixed input prefixes; an FP32 model then recomputes every complete prefix.
For each of the three probability/mass/normalized-log metrics, the budget is
the independently calibrated reference budget plus that input's measured
BF16/FP32 discrepancy. Engine probabilities never determine their own budget.
The same budget must reject half-normalization, temperature-two and wrong-context
faults. Excessive reference uncertainty fails explicitly rather than passing.

Every runtime numerical check retains all requests and positions at offsets
`0, 1024, 32768, 98304, 130560, 131584`, including short variable-length and
single-request cases. These are windows of one fixed stream, not statistically
independent datasets. Reference qualification also uses retained and fresh
reference-only corpora; 1,152 non-anchor comparisons supplement the canonical
cached-batch anchors. This empirical contract is not a universal BF16 error bound.

#### Isolated Runtime

The engine requires vLLM 0.31.0 and a compatible CUDA/ROCm/PyTorch installation.
Do not install vLLM over the existing SuperBench training environment.
Set `runtime_python` to the interpreter in its separate engine environment.
Set `reference_python` to a separate environment containing the pinned
[`official-code reference packages`](../../../dockerfile/requirements/deepseek-v2-lite-reference.txt).
The controller only imports standard-library worker helpers, not vLLM.

The optional
[`model runtime Dockerfile`](../../../dockerfile/deepseek-v2-lite-inference.dockerfile)
installs the Python SuperBench controller and its `sb` entrypoint in the
reference-compatible environment without replacing the engine's Transformers.
`/opt/vllm/bin/python` selects the engine interpreter;
`/opt/deepseek-reference/bin/python` selects the reference/controller interpreter.
The
[`controller lock`](../../../dockerfile/requirements/deepseek-v2-lite-controller.txt)
retains the base GPU Torch/torchvision installation; its regeneration command is
in the [lock inputs](../../../dockerfile/requirements/deepseek-v2-lite-controller.in).
This is an opt-in model runtime, not the standard compiled microbenchmark image.
CUDA/ROCm registration is not certification of every GPU/runtime combination.
The build checks both interpreters' framework versions and Torch/NumPy array
interoperability. Controller/reference package pins must not replace the engine's
own environment.

```bash
docker build -f dockerfile/deepseek-v2-lite-inference.dockerfile -t sb-v2lite-inference .
docker run --rm --gpus device=0 --shm-size 32g \
  -v "$PWD/v2lite-results:/results" sb-v2lite-inference exec \
  --config-file superbench/config/deepseek-v2-lite-inference.yaml \
  --output-dir /results/sb-output \
  -C superbench.benchmarks.deepseek-v2-lite-inference.parameters.cache_root=/results/model-cache
```

For a separately prepared native installation:
```bash
sb exec --config-file superbench/config/deepseek-v2-lite-inference.yaml
```

The command assumes SuperBench and both worker environments are already prepared
and accessible on the same host. In a container, mount the updated checkout
read-only and provide a writable, dedicated result/cache directory.

#### Correctness And Lifecycle

The engine exercises variable-length batched prompts, single requests and forced
EOS termination. Outputs must match the submitted prompt token IDs. A separate
unsharded official-code FP32 reference, with TF32 disabled, then checks
prefill and multiple decode steps on the same token prefixes, comparing cached
and complete-prefix logits across the vocabulary and sampled engine top-token
log probabilities and requires greedy token agreement. Sampled engine
probabilities do not constitute an exhaustive
comparison of the engine's full vocabulary logits or all expert routing paths.
Correctness, complete requests, finite timestamps and actual token counts must
all pass; a plausible generated sentence or process exit zero is insufficient.
Numerical records must contain the requested top-20 log probabilities, not only
the selected token. For long selected requests, every generated token is checked
against the cached FP32 reference; full-prefix recomputation checks the first,
second, middle and final positions. A position without full-prefix recomputation
is explicitly marked rather than reporting a fabricated zero cache error.

Engine model residency is BF16, with no quantized or offloaded fallback.
The router uses FP32 to match the official gate. NVIDIA TF32, vLLM FP32 matmul
precision and Triton FP32 arithmetic are explicitly controlled; these settings
do not turn the BF16 performance workload into full-FP32 inference.
The worker enables `VLLM_RAISE_ON_LOGIT_NANS` before importing vLLM, so NaN
logits fail the run even during measured requests that do not collect log
probabilities. Corrupted request statistics also prevent a successful report.
Inter-request prefix caching and chunked prefill are disabled for this initial
profile; within-request decode state is retained. Changing those policies or
using tensor/expert parallelism requires a separately named validation profile.

vLLM 0.31.0 reports engine-core monotonic timestamps and provides explicit worker
joining through its engine-core client. This version-specific access is
intentional: incompatible metrics or shutdown APIs fail rather than supplying
default timings. The engine is shut down before the independent reference is
loaded, so their full model copies do not simultaneously consume GPU memory.
The FP32 reference has approximately 62.8GB of parameter storage alone, so this
initial complete validation profile needs an 80GB-class GPU even if the BF16
engine could fit on a smaller device. That is a reference-profile requirement,
not a claim that the model intrinsically requires H100 or eight GPUs.

#### Timing And Metrics

Every invocation uses a new empty HF/Hub/Xet/module cache without deleting shared
caches. One `duration <3600s` covers model download, complete weight hashes,
loading, graph/kernel preparation, warmup, measured generation, reference checks,
result writing and worker teardown. The controller reserves cleanup time and
rejects watchdog exits. Image pulls, dependency installation and controller
installation are separate preparation steps, not certified by the model-cold
timer.

A dedicated Linux child supervisor reaps worker descendants even when they start
separate sessions. A successful launcher exit with surviving children is rejected.
Worker measurement-phase Unix timestamps allow external telemetry to exclude
setup, correctness and warmup; the timed generation durations use a monotonic
clock. CPU regression tests cover warmup exclusion and mixed EOS/decode counts.

| Metric | Unit | Scope |
| --- | --- | --- |
| `bf16_inference_step_time` | ms | Mean complete synchronized generation batch |
| `bf16_inference_output_tokens_per_second` | tokens/s | Actual outputs divided by measured batch time |
| `bf16_inference_request_prefill_time` | ms | Engine scheduled-to-first-token interval |
| `bf16_inference_ttft` | ms | Engine first-token latency, including frontend/queue effects |
| `bf16_inference_decode_tpot` | ms/token | First-to-last-token span divided by actual decode intervals |
| `successful_requests` | requests | Completed measured requests |
| `actual_output_tokens` | tokens | Actual measured output IDs, not the configured maximum |
| `end_to_end_seconds` | s | Complete worker launch through teardown/report validation |

Request prefill/decode timestamps are not isolated GPU kernel timings.
One-token EOS requests contribute to total tokens and latency but not to TPOT.
Keep cold-start time, steady per-GPU utilization and model quality separate;
GPU device busy time is not MFU.
