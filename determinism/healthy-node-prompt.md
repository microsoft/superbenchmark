# Healthy-Node Agent Prompt — Gold Baseline Generation

Paste the block below into the coding agent running on the **healthy (good) node's** VM.
Its only job is to generate and publish a trustworthy gold baseline of activation hashes
that the suspected-bad nodes compare against.

---

```
You are operating on the HEALTHY (good) node in a multi-node Silent Data Corruption (SDC)
detection experiment for SuperBench. Your single job: produce a trustworthy GOLD BASELINE
of activation hashes that the suspected-bad nodes will compare against. The bad nodes hunt
for SDC; this node only generates the reference. Work in /opt/superbench.

CONTEXT
- The detection feature lives on git branch `sdc-atp-clean` of the microsoft/superbenchmark repo.
- It records a deterministic per-checkpoint activation hash (Approach B) every check_frequency
  steps, plus a combined hash, into each run's results-summary.jsonl.
- A baseline is just ONE clean, deterministic run's results-summary.jsonl copied to
  new_tests_determinis_gold_data/baseline.json. The healthy node does NOT need the long
  multi-pass soak loop the bad nodes use — it only needs to (a) generate one baseline and
  (b) prove the node is self-consistent by running twice and confirming identical hashes.

STEP 1 — Get the code and install (editable):
  cd /opt/superbench
  git fetch origin
  git checkout sdc-atp-clean
  git pull origin sdc-atp-clean
  python3 -m pip install -e . --no-deps

STEP 2 — Verify the new per-checkpoint hash code is live (must print matching lines):
  grep -n "act_hash_ckpt" superbench/benchmarks/model_benchmarks/pytorch_base.py
  python3 -c "import superbench.benchmarks.model_benchmarks.pytorch_base as m; print(m.__file__)"
  # The printed path MUST be /opt/superbench/... (the editable install), not a site-packages path.

STEP 3 — Run TWO back-to-back 6h hash passes (generates the baseline AND self-validates the node).
  This uses the identical config the bad nodes use (seed 42, num_steps 5500, fp32, batch_size 24,
  seq_len 256, check_frequency 100, 4 ranks). Do NOT change any of these — hash comparison across
  nodes is only valid with identical config. Launch in the background so it survives SSH drop:
  mkdir -p temp/logs
  nohup determinism/run-loop.sh 2 determinism/llama2-7b-det-6h-hash.yaml > temp/logs/gold-loop.log 2>&1 &
  echo "launched PID $!"

  IMPORTANT: `sb run` MUST include `--host-list localhost` or it errors with
  "Must specify one of host_file or host_list." run-loop.sh already passes this.

STEP 4 — Monitor (do not poll aggressively; check periodically). Confirm health with:
  pgrep -fa "run-loop.sh"; pgrep -fa "sb run"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
  tail -n 20 temp/logs/gold-loop.log
  All 4 GPUs should be ~100% util. Each run is ~6h (~3867 ms/step x 5500 steps); two runs ~12h.

STEP 5 — When both runs finish, run-loop.sh auto-runs determinism/compare-hashes.py. The two
  runs MUST report "all activation hashes consistent across runs". If they DIVERGE, STOP — the
  healthy node is itself nondeterministic and is NOT safe to use as gold; report this immediately.

STEP 6 — Promote run 1's results to the gold baseline:
  MANIFEST=$(ls -t temp/logs/soak-*-outputs.txt | head -1)
  GOLD=$(head -1 "$MANIFEST")          # run 1's results-summary.jsonl
  echo "Using $GOLD"
  cp "$GOLD" new_tests_determinis_gold_data/baseline.json
  # Sanity-check it contains per-checkpoint hash keys:
  python3 -c "import json;d=json.load(open('new_tests_determinis_gold_data/baseline.json'));print('hash keys:',len([k for k in d if 'act_hash' in k]))"
  # Expect many keys: 4 combined (one per rank) + per-checkpoint act_hash_ckpt{i} x 4 ranks.

STEP 7 — Publish the new baseline so the bad nodes can pull it:
  git add new_tests_determinis_gold_data/baseline.json
  git commit -m "Regenerate gold baseline with per-checkpoint hashes (healthy node)"
  git push origin sdc-atp-clean

CONSTRAINTS / GOTCHAS
- Never modify the config values (seed, num_steps, check_frequency, dtype, batch_size, seq_len).
  Any difference invalidates cross-node hash comparison.
- The combined 18-digit hash is stored as float64 by the baseline pipeline and loses its lowest
  digits — this is EXPECTED and handled (compare-hashes.py normalizes through float64). Per-checkpoint
  hashes are reduced mod 1e15 so they stay float-exact.
- This node must complete its runs cleanly (return code 0, all 5500 steps). A crash/timeout means
  the baseline is incomplete — re-run.
- Do not delete or edit other nodes' output dirs.

REPORT BACK when done: the gold baseline commit hash, whether the two runs were self-consistent,
and the number of hash keys in baseline.json.
```
