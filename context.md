# Context: Per-Step SDC Quorum Detection

## Summary

This branch adds **per-step Silent Data Corruption (SDC) detection** to SuperBench,
porting NVIDIA's DNM6 (`llm_launcher/analysis.py`) exact-match quorum voting. Instead
of averaging periodic fingerprints into one scalar per run and comparing against a
baseline, it compares **every training step independently across participants** using
**bit-exact equality**, so a single corrupted step on one GPU is caught immediately and
attributed, rather than being diluted by averaging.

## Background

SuperBench previously recorded periodic determinism fingerprints (loss, activation mean)
at `check_frequency` intervals and reduced them to a mean per run. NVIDIA's approach
compares each step across nodes bit-for-bit; the largest exact-match value-group at a
step wins (unambiguous plurality) and any participant outside it is an outlier at that
step. This is strictly stronger: one bad step is flagged instead of averaged away, and
the exact divergent step and value are reported.

## What changed

| File | Change |
|------|--------|
| `superbench/analyzer/sdc_quorum.py` (new) | Core quorum module: `compute_quorum` (first-seen ordering, uniquely-largest-group wins, ties are ambiguous), `compute_quorum_by_rank` (compare same global_rank across nodes), rank-aware `extract_from_results`, `merge_results_files` helper, and text/JSON reporters + `run_sdc_check`. |
| `superbench/common/model_log_utils.py` | Added `record_step_fingerprint` to record every step's loss and activation mean (numeric, with NaN/Inf sentinels). |
| `superbench/benchmarks/model_benchmarks/pytorch_base.py` | Per-step storage, NCCL determinism env vars, and export of per-step loss/act_mean as a flat numeric `raw_data` series. |
| `superbench/cli/_commands.py`, `superbench/cli/_result_handler.py` | New `sb result sdc-check` command with output-format validation and non-zero exit on detection. |
| `superbench/analyzer/__init__.py` | Expose public API. |
| `examples/benchmarks/pytorch_deterministic_example.py` | Surface per-step raw data as JSON in the results summary consumed by `sb result sdc-check`. |
| `tests/analyzer/test_sdc_quorum.py` (new) | 30 unit tests for the quorum module. |

## Design decisions (gap fixes vs. the original plan)

- **Rank-aware comparison.** Participants are compared as `(node, rank)`, and voting is
  grouped **per rank** (`compute_quorum_by_rank`) so an outlier attributes to a specific
  `node.rankN` GPU. In SuperBench's DDP model benchmarks each rank trains on a different
  data shard, so only the *same* rank across nodes is directly comparable.
- **First-seen ordering.** Group and outlier ordering follows first appearance to match
  the NVIDIA reference output.
- **Storage format.** Per-step data is stored in `raw_data` as a flat `[step, value, ...]`
  numeric list (valid `List[List[Number]]`), not as a JSON string in `result` — the latter
  failed the model benchmark's `List[Number]` summarized-result validation and set
  `return_code=3` for every determinism run.
- **Merge helper + format validation** added; `sb result sdc-check` exits non-zero on
  detection for CI.

## Scope / limitations

- Detects and attributes **which participant (GPU/rank/node)** diverged and **at which
  step** — it does **not** localize the failing component (compute vs. memory vs.
  communication). Component localization is a separate stage using other tools
  (`nccl-tests`, `gpu-burn`, DCGM, `nvbandwidth`, etc.).
- Attribution needs **>= 3 comparable participants** per rank-group; with only 2 a
  divergent step is a 1-vs-1 tie and is correctly reported as ambiguous.
- Covers all PyTorch model benchmarks (BERT, CNN, GPT-2, LLaMA, LSTM, Mixtral) via the
  centralized `record_determinism_fingerprint` hook; `megatron_gpt3` uses a separate path
  and is not covered.

## How to run

```bash
# 1. Run the deterministic benchmark 3x (simulating 3 nodes)
for i in 0 1 2; do
  python examples/benchmarks/pytorch_deterministic_example.py \
    --model lstm --enable_determinism --deterministic_seed 42
done

# 2. Merge the runs with distinct node labels
python -c "
from superbench.analyzer.sdc_quorum import merge_results_files
import glob
files = sorted(glob.glob('outputs/*/results-summary.jsonl'))[:3]
merge_results_files(files, '/tmp/merged.jsonl', node_names=['node0','node1','node2'])
"

# 3. Run the check (PASS on clean runs; FAIL + exit 1 if any participant diverges)
sb result sdc-check -d /tmp/merged.jsonl --metric-pattern deterministic_loss_per_step
```

## Validation

- 30 SDC unit tests + 6 model-log tests pass; determinism benchmark tests: 12 passed / 2 skipped.
- End-to-end on 4x GB300: three identical deterministic LSTM runs -> `PASS`; injecting a
  fault on one run -> `FAIL` naming `node1.rank0` at the corrupted step (exit 1).
- The corruption in the end-to-end run was **injected** to exercise the detector; it
  validates the detection path, not a genuine hardware fault.

## Reference

Algorithm ported from the DNM6 image's `usr/local/llm_launcher/src/llm_launcher/analysis.py`
(`compute_quorum`). DNM6 = "Deterministic Nemotron-6", NVIDIA's determinism-locked GPU
SDC screening workload.
