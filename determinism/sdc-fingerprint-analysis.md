# Determinism Fingerprint vs. SDC Detection — Analysis

## Context

- Run: `model-benchmarks:llama2-7b-det` (~24h), output `outputs/2026-06-21_05-02-34/`
- Baseline: `/opt/superbench/determinism/baseline.json` (gold data)
- Diagnosis rule: `/opt/superbench/determinism/diagnosis-rule.yaml`
- Result: `diagnosis_summary.md` is **empty** → no metric deviated from baseline.

The node's `deterministic_loss` and `deterministic_act_mean` are **identical** to the
gold baseline:

```
deterministic_loss     = 0.00012105521727755136   (== gold, all 4 ranks)
deterministic_act_mean = -0.7710989933946858       (== gold, all 4 ranks)
```

This machine is *known* to have SDCs, yet the fingerprint check passes.

## Short answer

A passing fingerprint does **not** prove the node is SDC-free. `deterministic_loss`
and `deterministic_act_mean` are very lossy scalar reductions, so they only catch
corruption that (a) survives heavy averaging and (b) happens to land on the sampled
steps/tensors. Much real SDC slips through.

## What the metrics actually are (from code)

From `superbench/common/model_log_utils.py`:

- **loss** = `float(loss.detach().item())` — one scalar per checkpoint.
- **act_mean** = `logits[0].detach().float().mean().item()` — mean of the logits for
  **sample 0 only**.

From `superbench/benchmarks/model_benchmarks/pytorch_base.py` (`_add_deterministic_metrics_to_result`):

- The reported value is `statistics.mean(filtered_values)` — the **mean across all
  checkpoints**.
- Checkpoints are taken only every `check_frequency` (= 1000) steps.

So each reported number collapses billions of tensor elements → one scalar → averaged
again over ~23 checkpoints → one number.

## Why SDC can be invisible to loss / act_mean

1. **Averaging cancels errors.** `act_mean` is a *mean* over the whole logits tensor.
   Sparse bit-flips (+x here, −x there) average toward zero. Mean is the worst statistic
   for sparse corruption; `max`/`min`/`sum-of-abs`/checksum would be far more sensitive.

2. **Only sample 0, only logits.** SDC affecting other samples, other layers, gradients,
   optimizer state, or internal activations never reaches this metric unless it
   propagates to sample 0's final logits.

3. **Sparse temporal sampling.** Fingerprints are recorded every 1000 steps. Transient
   SDC (cosmic-ray flip, marginal SM miscomputing occasionally) between checkpoints
   leaves no trace if the model recovers by the next checkpoint.

4. **Determinism settings mask hardware nondeterminism.** Fixed seed, deterministic
   algorithms, and `CUBLAS_WORKSPACE_CONFIG` remove *software/algorithmic*
   nondeterminism (intended), but also push kernels onto deterministic code paths that
   may not exercise the faulty hardware unit.

5. **Self-healing dynamics.** Training contracts toward a minimum (loss ~1e-7..1e-4,
   essentially converged). Small perturbations get pulled back, so the loss fingerprint
   is robust to exactly the small corruption SDC tends to produce.

6. **fp32 + tiny magnitudes.** Values are tiny (loss ~1.2e-4). A low-order-bit SDC may
   not change the value at the compared precision, especially after float→mean rounding.

## What this means here

The fingerprint match tells us: *under fully-deterministic settings, this node
reproduces the reference's loss and sample-0 logit mean.* It does **not** prove the node
is SDC-free. The known SDC may be sparse/transient, affect units/paths not exercised by
this config, or manifest in components the fingerprint never inspects.

## How to actually catch it

The determinism fingerprint is a coarse screen, not an SDC detector. To catch real SDC:

- **More sensitive fingerprints** (code change): hash/checksum of full logits instead of
  mean; track `max`/`min`/`std`/L2-norm; fingerprint *all* samples and intermediate
  activations; record per-step (not 1000-step) values.
- **Dedicated SDC stress tests:**
  - `gpu-burn` (in `bin/`) — long matrix-multiply with result verification.
  - `nvbandwidth` / `gpu-copy-bw` with data verification.
  - NVIDIA DCGM diagnostics (`dcgmi diag -r 3/4`) and hardware health checks.
  - cuBLAS/cuBLASLt GEMM benchmarks with golden comparison over many iterations.
