# Handover: Deterministic SDC Detection — Max/Min Fingerprint Feature

Purpose: enable better Silent Data Corruption (SDC) detection in the
`model-benchmarks:llama2-7b-det` determinism run by adding non-diluting
`_max`/`_min` fingerprint metrics, and run a short (~6h) fixed-step validation.

Apply ALL steps below on each of the 2 target machines. Paths assume the
SuperBench checkout is at `/opt/superbench`.

---

## Background / why

The determinism fingerprint reports `deterministic_loss` and
`deterministic_act_mean` as a **mean over all checkpoints** (e.g. 23 points for a
24h run). A single corrupted checkpoint (potential SDC) is diluted by averaging
(~23x at check_frequency=1000, ~230x at 100), so transient SDC can be invisible.
`max`/`min` are NOT diluted by averaging — a single bad checkpoint still surfaces.

Additional note: the 24h run was **duration-based** (`duration: 86400`,
`num_steps: 0`). Different-speed machines reach different step counts, so the
fingerprint sequence length differs and reductions are not comparable. For a
valid comparison we switch the test to a **fixed step count**.

---

## CHANGE 1 — Code: add max/min fingerprint metrics

File: `superbench/benchmarks/model_benchmarks/pytorch_base.py`
Method: `_add_deterministic_metrics_to_result` (~line 181)

Replace this block:

```python
                    # Add summarized result (mean of checkpointed values)
                    filtered_values = [v for v in values if v is not None]
                    if filtered_values:
                        self._result.add_result(metric_name, statistics.mean(filtered_values))
                    else:
                        # No valid (non-None) values recorded; record NaN to avoid StatisticsError
                        self._result.add_result(metric_name, float('nan'))
```

with:

```python
                    # Add summarized result (mean of checkpointed values)
                    filtered_values = [v for v in values if v is not None]
                    if filtered_values:
                        self._result.add_result(metric_name, statistics.mean(filtered_values))
                        # Add max/min of checkpointed values. Unlike the mean, these are not
                        # diluted by averaging, so a single corrupted checkpoint (potential SDC)
                        # still surfaces regardless of how many checkpoints were recorded.
                        self._result.add_result(f'{metric_name}_max', max(filtered_values))
                        self._result.add_result(f'{metric_name}_min', min(filtered_values))
                    else:
                        # No valid (non-None) values recorded; record NaN to avoid StatisticsError
                        self._result.add_result(metric_name, float('nan'))
```

New metrics produced per rank:
`deterministic_loss_max/min`, `deterministic_act_mean_max/min`.

---

## CHANGE 2 — Deploy the code (editable install)

`sb` runs from the installed package, not the source tree. Reinstall editable so
the change takes effect (and future edits apply automatically):

```bash
cd /opt/superbench && python3 -m pip install -e . --no-deps
```

Verify the change is live:

```bash
python3 -c "import superbench.benchmarks.model_benchmarks.pytorch_base as m; print(m.__file__)"
grep -n "_max'" /opt/superbench/superbench/benchmarks/model_benchmarks/pytorch_base.py
```

Expect `m.__file__` to point at `/opt/superbench/...` and the grep to show the
`_max` line.

---

## CHANGE 3 — 6h fixed-step test config (NEW FILE)

File: `determinism/llama2-7b-det-6h.yaml`

```yaml
# SuperBench config: 6h llama2-7b determinism (SDC) test run.
# Uses a FIXED step count (not duration) so the fingerprint sequence length is
# reproducible across machines, which makes the per-checkpoint max/min/mean
# determinism metrics comparable. ~5500 steps ≈ 6h at ~3.87 s/step on this node.
version: v0.11
superbench:
  enable:
  - model-benchmarks:llama2-7b-det
  monitor:
    enable: false
  var:
    default_local_mode:
      modes:
      - name: local
        proc_num: 4
        prefix: CUDA_VISIBLE_DEVICES={proc_rank}
        parallel: yes
  benchmarks:
    model-benchmarks:llama2-7b-det:
      modes:
      - name: local
        proc_num: 4
        prefix: CUDA_VISIBLE_DEVICES={proc_rank}
        parallel: yes
      frameworks: [pytorch]
      timeout: 25200            # >= ~6h + setup overhead (6h = 21600s)
      models:
      - llama2-7b
      parameters:
        duration: 0             # disable wall-clock cap; use fixed steps instead
        num_warmup: 32
        num_steps: 5500         # ~6h at ~3.87 s/step (reproducible fingerprint length)
        sample_count: 8192
        batch_size: 24
        precision: [float32]
        model_action: [train]
        pin_memory: yes
        num_workers: 0
        seq_len: 256
        enable_determinism: true
        deterministic_seed: 42
        check_frequency: 100    # ~55 checkpoints -> exercises max/min metrics well
```

IMPORTANT — step count is hardware-dependent:
`num_steps ≈ (6 * 3600) / step_time_seconds`. ~3.87 s/step on the reference node
gives ~5500. If a target machine is faster/slower, recompute from one short
warm-up run's `fp32_train_step_time` (ms) so the run stays near 6h AND all
machines compare at the SAME `num_steps`. For a valid cross-machine determinism
comparison, all machines MUST use the SAME `num_steps`.

---

## CHANGE 4 — Diagnosis rule: include new metrics (EDIT)

File: `determinism/diagnosis-rule.yaml`

Notes:
- The `SDC-Fingerprint` rule now lists the `_max`/`_min` metrics explicitly.
- `deterministic_config_num_steps` was REMOVED from the `SDC-Config` variance
  rule earlier because the `variance` op divides by the baseline; a baseline of 0
  (duration-based runs) caused a divide-by-zero error. With fixed `num_steps`
  (Change 3) the baseline is non-zero, so you MAY re-add it if desired.

Full file:

```yaml
superbench:
  rules:
    deterministic_rule:
      function: variance
      criteria: "lambda x: x != 0"
      categories: SDC-Fingerprint
      metrics:
        - model-benchmarks:.*/deterministic_loss.*
        - model-benchmarks:.*/deterministic_loss_max.*
        - model-benchmarks:.*/deterministic_loss_min.*
        - model-benchmarks:.*/deterministic_act_mean.*
        - model-benchmarks:.*/deterministic_act_mean_max.*
        - model-benchmarks:.*/deterministic_act_mean_min.*
        - model-benchmarks:.*/deterministic_check_count.*

    deterministic_config_rule:
      function: variance
      criteria: "lambda x: x != 0"
      categories: SDC-Config
      metrics:
        - model-benchmarks:.*/deterministic_config_batch_size.*
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

---

## CHANGE 5 — Summary rule: include new metrics (EDIT)

File: `gb300_summary_rules.yaml`, the `model-benchmarks-deterministic` rule.

The `Deterministic` category metrics list should be:

```yaml
      metrics:
        - model-benchmarks:.*/deterministic_loss.*
        - model-benchmarks:.*/deterministic_loss_max.*
        - model-benchmarks:.*/deterministic_loss_min.*
        - model-benchmarks:.*/deterministic_act_mean.*
        - model-benchmarks:.*/deterministic_act_mean_max.*
        - model-benchmarks:.*/deterministic_act_mean_min.*
        - model-benchmarks:.*/deterministic_check_count.*
        - model-benchmarks:.*/deterministic_step.*
        - model-benchmarks:.*/deterministic_config_.*
        - model-benchmarks:.*/return_code.*
```

---

## RUN WORKFLOW (per machine)

1. Apply Changes 1–5, then editable-install (Change 2).
2. Run the 6h test:
   ```bash
   cd /opt/superbench
   sb run --no-docker -c determinism/llama2-7b-det-6h.yaml
   ```
   Output goes to `outputs/<datetime>/`. Results file:
   `outputs/<datetime>/results-summary.jsonl`.

3. On the KNOWN-GOOD machine, generate a fresh baseline (the old
   `baseline.json` predates the `_max`/`_min` keys, so it MUST be regenerated):
   ```bash
   sb result generate-baseline \
     --data-file outputs/<good-run>/results-summary.jsonl \
     --diagnosis-rule-file determinism/diagnosis-rule.yaml \
     --summary-rule-file gb300_summary_rules.yaml \
     --output-dir determinism/baseline-6h
   ```
   Share the resulting `baseline.json` with the other machines.

4. On each machine, run the determinism (SDC) diagnosis against the gold baseline:
   ```bash
   sb result diagnosis \
     --data-file outputs/<this-run>/results-summary.jsonl \
     --rule-file determinism/diagnosis-rule.yaml \
     --baseline-file determinism/baseline-6h/baseline.json \
     --output-dir outputs/<this-run>/determinism-check \
     --output-file-format md \
     --output-all
   ```
   - Empty `diagnosis_summary.md` / `diagnosis/accept: true` => matched gold.
   - Any rows / `accept: false` under `SDC-Fingerprint` => deviation from gold
     (potential SDC). `_max`/`_min` deviations indicate a single bad checkpoint
     that the mean would have hidden.

5. (Optional) Readable summary report:
   ```bash
   sb result summary \
     --data-file outputs/<this-run>/results-summary.jsonl \
     --rule-file gb300_summary_rules.yaml \
     --output-dir outputs/<this-run>/summary \
     --output-file-format md
   ```

---

## GOTCHAS / VALIDATION CHECKLIST

- [ ] Editable install done; `sb` resolves to `/opt/superbench` source.
- [ ] All machines use the SAME `num_steps` (recompute per-hardware step time so
      wall-time ≈ 6h, but keep `num_steps` identical across machines for a valid
      comparison).
- [ ] Baseline REGENERATED with the new code (must contain `_max`/`_min` keys);
      do not reuse the old `baseline.json`.
- [ ] `deterministic_config_num_steps` left OUT of the variance config rule, OR
      re-added only if `num_steps > 0` for all compared runs (fixed-step avoids
      the divide-by-zero).
- [ ] `check_frequency: 100` => ~55 checkpoints; enough to exercise max/min.
- [ ] Determinism comparison is a coarse screen, not a guaranteed SDC detector.
      Shorter runs reduce exposure to intermittent SDC; pair with dedicated
      stress tests (gpu-burn, GEMM verification, `dcgmi diag`) for real hunting.

---

## FILES TOUCHED (summary)

| File | Action |
|------|--------|
| `superbench/benchmarks/model_benchmarks/pytorch_base.py` | EDIT: add `_max`/`_min` metrics |
| `determinism/llama2-7b-det-6h.yaml` | NEW: 6h fixed-step test config |
| `determinism/diagnosis-rule.yaml` | EDIT: add `_max`/`_min` metrics (num_steps removed from config rule) |
| `gb300_summary_rules.yaml` | EDIT: add `_max`/`_min` metrics |
| `determinism/sdc-fingerprint-analysis.md` | reference: why mean hides SDC |
