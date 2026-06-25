#!/usr/bin/env python3
"""Compare activation hashes across back-to-back runs (and against a gold baseline).

Reads a manifest file listing one results-summary.jsonl path per line (produced by
run-loop.sh) and checks whether the activation-hash metrics are identical across all
runs. Any difference -- combined ``deterministic_act_hash`` or any per-checkpoint
``deterministic_act_hash_ckpt{i}`` -- is flagged as a potential SDC.

Two checks:
  1. Self-consistency: every run must produce the SAME hashes as the first run. This
     needs no gold baseline -- divergence between two runs on the same node is itself
     proof of nondeterminism / SDC.
  2. Gold (optional): each run's hashes must match the provided gold baseline.json.
     NOTE: a gold baseline generated before per-checkpoint hashes existed will only
     contain the combined ``deterministic_act_hash`` keys; per-checkpoint keys are then
     skipped for the gold check but still used for self-consistency.

Exit code 0 = all hashes consistent; 1 = at least one divergence found.
"""
import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def hash_keys(d):
    return {k: v for k, v in d.items() if 'act_hash' in k}


def norm(v):
    """Normalize a hash value through float64 so comparisons match the baseline's storage.

    The combined ``deterministic_act_hash`` is an 18-digit int that cannot be represented
    exactly in float64 (the type the baseline/diagnosis pipeline stores). Comparing a run's
    exact int against the gold's float-rounded value would yield a false mismatch, so we
    round both sides identically. Per-checkpoint hashes are reduced mod 1e15 (< 2**53) and
    are therefore float-exact, so this is a no-op for them.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest', help='file listing results-summary.jsonl paths, one per line')
    ap.add_argument('--gold', help='gold baseline.json to compare against', default=None)
    args = ap.parse_args()

    with open(args.manifest) as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    if not paths:
        print('[compare] manifest is empty; nothing to compare')
        return 0

    runs = [(p, hash_keys(load(p))) for p in paths]
    print(f'[compare] {len(runs)} run(s); {len(runs[0][1])} hash metrics in run 1')

    diverged = False

    # 1) Self-consistency against run 1.
    base_path, base = runs[0]
    for path, cur in runs[1:]:
        keys = set(base) | set(cur)
        for k in sorted(keys):
            bv, cv = base.get(k), cur.get(k)
            if norm(bv) != norm(cv):
                diverged = True
                print(f'[SDC] mismatch vs run1 in {path}\n        key={k}\n        run1={bv} this={cv}')

    # 2) Gold comparison (combined hash always; per-checkpoint only if present in gold).
    if args.gold:
        gold = hash_keys(load(args.gold))
        gck = set(gold)
        for path, cur in runs:
            for k in sorted(gck):
                if norm(gold[k]) != norm(cur.get(k)):
                    diverged = True
                    print(f'[SDC] gold mismatch in {path}\n        key={k}\n        gold={gold[k]} this={cur.get(k)}')
        only_run = set(runs[0][1]) - gck
        if only_run:
            print(f'[compare] note: {len(only_run)} hash metric(s) not in gold (e.g. per-checkpoint) '
                  f'-> checked for self-consistency only')

    if diverged:
        print('[compare] RESULT: DIVERGENCE DETECTED -- potential SDC')
        return 1
    print('[compare] RESULT: all activation hashes consistent across runs'
          + (' and gold' if args.gold else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
