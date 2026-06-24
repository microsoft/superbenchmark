# SDC Detection — Multi-Node Test Plan

Purpose: coordinate a structured Silent Data Corruption (SDC) detection effort
across the available nodes. The 24h mean-based determinism fingerprint failed to
surface known SDCs, so this plan validates stronger detectors against ground
truth: **1 known-good node** (gold baseline) and **3 known-bad nodes** (each runs
a different detector so we can rank which one reliably catches the known faults).

All nodes use the SuperBench checkout at `/opt/superbench` on branch
`sdc-atp-clean`. Paths below are relative to that directory.

---

## Why this layout

- The **good node** produces a trustworthy gold baseline and acts as the
  false-positive check (a good detector must NOT flag it).
- The **bad nodes** are positive cases — we *know* a detector should trip on them.
  Running a different detector on each lets us measure true-positive rate and
  rank detectors by sensitivity and cost.

The faults are suspected to be inside a single GPU (NVIDIA's analysis moved to
IST), i.e. the **SM/compute path** or the **HBM interface** — not NVLink. The
detectors below target the compute path. (NVLink would require collective
`*_perf` tests, which are out of scope for this round.)

---

## Prerequisites (every node)

```bash
cd /opt/superbench
git fetch origin
git checkout sdc-atp-clean
python3 -m pip install -e . --no-deps   # only if SuperBench is installed editable
```

**Critical — keep these IDENTICAL across all 4 nodes**, or fingerprints are not
comparable and a "mismatch" may be config, not SDC:

| Parameter | Value |
|---|---|
| `num_steps` | `5500` (do NOT change per machine — wall-clock differs, step count must match) |
| `deterministic_seed` | `42` |
| `batch_size` | `24` |
| `seq_len` | `256` |
| `precision` | `[float32]` |
| `check_frequency` | `100` |

> `num_steps: 5500` is tuned to ~3.87 s/step on node0-atp (~6h). Other nodes will
> take a different amount of wall-clock time for the same 5500 steps — that is
> expected and fine. The step count is what must stay equal.

---

## Node assignments

| Node | Role | Config to run | Detector |
|---|---|---|---|
| **GOOD node** | Gold baseline + false-positive check | `determinism/llama2-7b-det-6h-hash.yaml` AND `…-chunks.yaml` | Generates reference for both A and B |
| **BAD node 1** | Approach B (exact hash) | `determinism/llama2-7b-det-6h-hash.yaml` | Bitwise activation hash — catches any divergence, incl. tiny in-range bit-flips |
| **BAD node 2** | Approach A (chunk checksums) | `determinism/llama2-7b-det-6h-chunks.yaml` | Per-chunk activation checksums — localizes a corruption to one segment |
| **BAD node 3** | Max/min (already shipped) | `determinism/llama2-7b-det-6h.yaml` | Non-diluting max/min of loss + act_mean (cheapest detector; the bar to beat) |

Rationale: the good node must run **both** A and B configs so each bad node has a
matching gold reference. Bad node 3 is the control for "does our cheap existing
improvement already catch it, or do we need the heavier hash?"

---

## Workflow

### Step 1 — Gold baseline (GOOD node first)

Run both detector configs on the good node:

```bash
sb run --no-docker --host-list localhost -c determinism/llama2-7b-det-6h-hash.yaml
sb run --no-docker --host-list localhost -c determinism/llama2-7b-det-6h-chunks.yaml
```

> `--host-list localhost` is REQUIRED for a local run. Without it `sb run` fails
> with `Must specify one of host_file or host_list.`

Generate a baseline from each good run (use the matching results-summary.jsonl):

```bash
sb result generate-baseline \
  --data-file outputs/<good-hash-run>/results-summary.jsonl \
  --diagnosis-rule-file determinism/diagnosis-rule.yaml \
  --summary-rule-file gb300_summary_rules.yaml \
  --output-dir determinism/baseline-6h-hash

sb result generate-baseline \
  --data-file outputs/<good-chunks-run>/results-summary.jsonl \
  --diagnosis-rule-file determinism/diagnosis-rule.yaml \
  --summary-rule-file gb300_summary_rules.yaml \
  --output-dir determinism/baseline-6h-chunks
```

Distribute the resulting `baseline.json` files to the matching bad nodes.

> The old `baseline.json` from the 24h run is stale — it predates the
> `_max`/`_min`/`act_chunk`/`act_hash` keys. Do NOT reuse it.

### Step 2 — Run detectors (BAD nodes, in parallel)

Each bad node runs only its assigned config. Run **looped and under thermal
load** — intermittent faults need many passes and heat:

```bash
# Example loop wrapper (adjust count as needed)
for i in $(seq 1 5); do
  sb run --no-docker --host-list localhost -c determinism/llama2-7b-det-6h-hash.yaml   # node's own config
done
```

### Step 3 — Diagnose each bad node against gold

```bash
sb result diagnosis \
  --data-file outputs/<bad-run>/results-summary.jsonl \
  --rule-file determinism/diagnosis-rule.yaml \
  --baseline-file determinism/baseline-6h-hash/baseline.json \
  --output-file-format json
```

(Use the chunks baseline on bad node 2, the original 6h baseline on bad node 3.)

### Step 4 — Score the detectors

For each detector record:

- [ ] Did it flag the bad node? (true positive)
- [ ] Did the GOOD node stay clean under the same config? (no false positive)
- [ ] How many passes / hours until it tripped?

**Winner** = catches all assigned bad nodes, stays clean on the good node, trips
fastest and cheapest. Deploy that detector fleet-wide.

---

## What each detector emits

| Detector | Metrics in results-summary.jsonl |
|---|---|
| Max/min | `deterministic_loss_max/min`, `deterministic_act_mean_max/min` |
| Approach A (chunks) | `deterministic_act_chunk<i>_max/min` (per chunk index) |
| Approach B (hash) | `deterministic_act_hash` (one combined hash per run) |

The diagnosis flags any non-zero variance from baseline. For the hash, any change
at all (value or order of checkpoints) changes the combined hash → flagged.

---

## Gotchas checklist

- [ ] All nodes on branch `sdc-atp-clean` with editable install refreshed.
- [ ] `num_steps`, seed, batch_size, seq_len, precision IDENTICAL everywhere.
- [ ] Gold baseline regenerated with THIS branch's code (not the old 24h one).
- [ ] Good node = same GPU model / driver / library stack as bad nodes. If not,
      the gold baseline is invalid — fall back to self-consistency (compare each
      node to its own repeated runs) instead of cross-node gold.
- [ ] Bad nodes run looped + hot; do not disqualify a detector after one clean run.
- [ ] NVLink not covered here — escalate to collective `*_perf` tests if all
      single-GPU detectors come back clean.

---

## Branch / files reference

| File | Role |
|---|---|
| `determinism/llama2-7b-det-6h.yaml` | Base 6h config (max/min metrics) |
| `determinism/llama2-7b-det-6h-hash.yaml` | Approach B config (bitwise hash) |
| `determinism/llama2-7b-det-6h-chunks.yaml` | Approach A config (per-chunk checksums) |
| `determinism/diagnosis-rule.yaml` | SDC diagnosis rules (incl. chunk/hash) |
| `gb300_summary_rules.yaml` | Summary report rules |
| `superbench/common/model_log_utils.py` | Fingerprint recorders (loss, act_mean, chunks, hash) |
| `superbench/benchmarks/model_benchmarks/pytorch_base.py` | Metric reduction + `--fingerprint_chunks` / `--fingerprint_hash` args |
