# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SDC Quorum Voting -- per-step exact-match comparison across nodes/GPUs.

This module implements NVIDIA-style quorum voting for Silent Data Corruption
(SDC/SDE) detection. Instead of comparing aggregated means against a baseline,
it compares per-step fingerprint values across all participants using bit-exact
equality, identifying the outlier participant(s) that produce a different value
at any step.

The comparison logic (ported from NVIDIA's llm_launcher ``analysis.py``):
  1. For each step, group all participants by their exact value.
  2. The uniquely largest group wins (unambiguous plurality).
  3. Any participant NOT in the winning group is an outlier at that step.
  4. Ties (two or more groups sharing the largest size) -> ambiguous (no verdict).

Comparison axis
---------------
The correct axis is *the same logical worker across identical runs*. In a
SuperBench distributed model benchmark each rank trains on a different data
shard, so rank-vs-rank comparison within a single node is meaningless. Instead
we compare the **same global rank across nodes** (the ``DistributedSampler``
shards deterministically by rank+seed, so rank ``k`` on every node sees the
same data). ``extract_from_results`` therefore emits one participant per
``(node, rank)`` and ``run_sdc_check`` groups the quorum by rank before voting,
so an outlier is attributed to a specific ``node.rankN`` (i.e. a specific GPU).

This is strictly stronger than mean-based variance comparison because:
  - A single corrupted step is caught immediately (not diluted by averaging).
  - NaN/Inf values are handled as sentinels (automatic outliers).
  - The exact step and the exact GPU of corruption are identified.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from superbench.common.utils import logger

# Sentinel prefix for NaN/Inf anomalies. A sentinel string never equals a float,
# so the participant automatically becomes an outlier at that step.
_ANOMALY_SENTINEL_PREFIX = 'anomaly:'

# Matches a trailing ``_rank<N>`` suffix on a metric key.
_RANK_SUFFIX_RE = re.compile(r'_rank(\d+)$')


def _anomaly_sentinel(detail: str) -> str:
    """Create a sentinel value for a NaN/Inf anomaly."""
    return f'{_ANOMALY_SENTINEL_PREFIX}{detail}'


def is_anomaly_value(value: Any) -> bool:
    """Check if a value is an anomaly sentinel."""
    return isinstance(value, str) and value.startswith(_ANOMALY_SENTINEL_PREFIX)


@dataclass
class LossExtraction:
    """Per-step loss/activation series for one participant.

    Attributes:
        node_name: Identifier for the node (hostname or logical node id).
        losses: Mapping of step_number -> value (float or sentinel string).
        anomaly_steps: Steps where NaN/Inf was detected.
        rank: Global rank of this participant, or None if not rank-partitioned.
    """
    node_name: str
    losses: Dict[int, Any] = field(default_factory=dict)
    anomaly_steps: List[int] = field(default_factory=list)
    rank: Optional[int] = None

    @property
    def identity(self) -> str:
        """Stable identity used in quorum grouping and reports.

        For rank-partitioned data this is ``<node>.rank<N>`` so an outlier maps
        to a specific GPU; otherwise it is just the node name (keeps parity with
        single-series usage and the reference test vectors).
        """
        if self.rank is None:
            return self.node_name
        return f'{self.node_name}.rank{self.rank}'


@dataclass
class QuorumGroup:
    """A group of participants that produced the same value at a given step."""
    value: Any
    node_names: List[str] = field(default_factory=list)


@dataclass
class QuorumStep:
    """Quorum result for a single step.

    Attributes:
        step: The step number.
        groups: All value-groups at this step, in first-seen order.
        majority_value: The value produced by the winning group (None if ambiguous).
        majority_nodes: Participants in the winning group.
        outlier_nodes: Participants NOT in the winning group (candidates for SDC).
        missing_nodes: Participants that have no value for this step.
        ambiguous: True if no uniquely largest group exists (tie at the top).
    """
    step: int
    groups: List[QuorumGroup] = field(default_factory=list)
    majority_value: Any = None
    majority_nodes: List[str] = field(default_factory=list)
    outlier_nodes: List[str] = field(default_factory=list)
    missing_nodes: List[str] = field(default_factory=list)
    ambiguous: bool = False


@dataclass
class QuorumResult:
    """Aggregate quorum result across all steps (and ranks).

    Attributes:
        total_steps: Number of distinct steps compared.
        total_participants: Number of participants compared.
        rank_count: Number of distinct rank-groups voted independently.
        steps: Per-step quorum details (populated for a single rank-group).
        has_outliers: True if any participant was an outlier at any step.
        outlier_summary: {identity: [outlier steps]} where identity is a GPU id.
        ambiguous_steps: Distinct steps where no verdict could be made.
        divergence_detail: {identity: first-divergence info} for reporting.
    """
    total_steps: int = 0
    total_participants: int = 0
    rank_count: int = 1
    steps: Dict[int, QuorumStep] = field(default_factory=dict)
    has_outliers: bool = False
    outlier_summary: Dict[str, List[int]] = field(default_factory=dict)
    ambiguous_steps: List[int] = field(default_factory=list)
    divergence_detail: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def compute_quorum(extractions: Sequence[LossExtraction]) -> QuorumResult:
    """Per-step exact-match quorum voting across a set of participants.

    For each step, groups participants by their exact value in first-seen order.
    The uniquely largest group wins; all other participants are outliers at that
    step. A tie at the top (two or more groups sharing the largest size) is
    ambiguous. This is the core algorithm ported from NVIDIA's ``analysis.py``.

    All participants passed here are assumed to be directly comparable (same
    rank / same data). Use :func:`run_sdc_check` (or group by rank yourself)
    before calling this on distributed results.

    Args:
        extractions: Participants to compare (one per node, or one per node/rank).

    Returns:
        QuorumResult with per-step verdicts and outlier summary.
    """
    if not extractions:
        return QuorumResult()

    all_steps = set()
    for ext in extractions:
        all_steps.update(ext.losses.keys())

    ordered_ids = [ext.identity for ext in extractions]
    id_map = {ext.identity: ext for ext in extractions}

    result = QuorumResult(total_steps=len(all_steps), total_participants=len(extractions))

    for step in sorted(all_steps):
        # Group participants by exact value, preserving first-seen ordering so
        # group and outlier ordering are deterministic and match the reference.
        value_groups: Dict[Any, List[str]] = {}
        order: List[Any] = []
        missing_nodes: List[str] = []

        for ident in ordered_ids:
            ext = id_map[ident]
            if step not in ext.losses:
                missing_nodes.append(ident)
                continue
            val = ext.losses[step]
            # Exact equality: floats compared bit-for-bit via Python ==.
            # Sentinel strings never equal floats -> automatic outlier group.
            if val not in value_groups:
                value_groups[val] = []
                order.append(val)
            value_groups[val].append(ident)

        groups = [QuorumGroup(value=v, node_names=list(value_groups[v])) for v in order]
        qs = QuorumStep(step=step, groups=groups, missing_nodes=missing_nodes)

        if not groups:
            qs.ambiguous = True
        else:
            top_size = max(len(g.node_names) for g in groups)
            top_groups = [g for g in groups if len(g.node_names) == top_size]
            if len(top_groups) == 1:
                # Unambiguous plurality: the uniquely largest group wins.
                winner = top_groups[0]
                qs.majority_value = winner.value
                qs.majority_nodes = list(winner.node_names)
                for g in groups:
                    if g is not winner:
                        qs.outlier_nodes.extend(g.node_names)
            else:
                # Tie at the top -> ambiguous, no verdict for this step.
                qs.ambiguous = True

        result.steps[step] = qs

        for ident in qs.outlier_nodes:
            result.outlier_summary.setdefault(ident, []).append(step)
        if qs.ambiguous:
            result.ambiguous_steps.append(step)

    result.has_outliers = len(result.outlier_summary) > 0
    _populate_divergence_detail(result)
    return result


def _populate_divergence_detail(result: QuorumResult) -> None:
    """Fill in first-divergence info for each outlier from ``result.steps``."""
    for ident, steps in result.outlier_summary.items():
        first_step = steps[0]
        qs = result.steps.get(first_step)
        node_val = None
        if qs is not None:
            for g in qs.groups:
                if ident in g.node_names:
                    node_val = g.value
                    break
        result.divergence_detail[ident] = {
            'first_outlier_step': first_step,
            'first_outlier_value': node_val,
            'majority_value_at_first_step': qs.majority_value if qs is not None else None,
            'outlier_step_count': len(steps),
            'all_outlier_steps': steps,
        }


def compute_quorum_by_rank(extractions: Sequence[LossExtraction]) -> QuorumResult:
    """Group participants by rank, vote within each rank, and combine.

    This is the distributed-safe entry point: rank ``k`` on node A is only ever
    compared against rank ``k`` on nodes B, C, ... (never against rank ``k+1``),
    so an outlier is attributed to a specific ``node.rankN`` GPU.

    If no participant carries a rank (all ``rank is None``) this degenerates to a
    single call to :func:`compute_quorum`.

    Args:
        extractions: Participants, each tagged with an optional rank.

    Returns:
        Combined QuorumResult across all rank-groups.
    """
    if not extractions:
        return QuorumResult()

    by_rank: Dict[Optional[int], List[LossExtraction]] = defaultdict(list)
    for ext in extractions:
        by_rank[ext.rank].append(ext)

    # Single group (no rank partitioning) -> plain quorum, keeps ``steps`` detail.
    if len(by_rank) == 1 and next(iter(by_rank)) is None:
        return compute_quorum(extractions)

    combined = QuorumResult(total_participants=len(extractions), rank_count=len(by_rank))
    all_steps = set()
    ambiguous = set()
    for rank in sorted(by_rank, key=lambda r: (r is not None, r)):
        group = by_rank[rank]
        # A rank-group with a single participant cannot be voted on; skip voting
        # but still count its steps so total_steps reflects coverage.
        sub = compute_quorum(group)
        all_steps.update(sub.steps.keys())
        ambiguous.update(sub.ambiguous_steps)
        for ident, steps in sub.outlier_summary.items():
            combined.outlier_summary[ident] = steps
        combined.divergence_detail.update(sub.divergence_detail)

    combined.total_steps = len(all_steps)
    combined.ambiguous_steps = sorted(ambiguous)
    combined.has_outliers = len(combined.outlier_summary) > 0
    return combined


def _coerce_step_value(step_str: Any, value: Any, losses: Dict[int, Any], anomaly_steps: List[int]) -> None:
    """Normalize one (step, value) pair into ``losses`` with sentinel handling."""
    step = int(step_str)
    if value is None:
        losses[step] = _anomaly_sentinel('nan')
        anomaly_steps.append(step)
    elif isinstance(value, str) and value.startswith(_ANOMALY_SENTINEL_PREFIX):
        losses[step] = value
        anomaly_steps.append(step)
    else:
        try:
            fval = float(value)
            if fval != fval:  # NaN
                losses[step] = _anomaly_sentinel('nan')
                anomaly_steps.append(step)
            elif abs(fval) == float('inf'):
                losses[step] = _anomaly_sentinel('inf')
                anomaly_steps.append(step)
            else:
                losses[step] = fval
        except (TypeError, ValueError):
            losses[step] = _anomaly_sentinel('parse_error')
            anomaly_steps.append(step)


def _parse_per_step_value(raw: Any) -> Optional[Dict[str, Any]]:
    """Unwrap a metric value (list-wrapped and/or JSON-encoded) into a dict."""
    val = raw
    if isinstance(val, list):
        val = val[0] if val else None
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return None
    if isinstance(val, dict):
        return val
    return None


def extract_from_results(
    results_file: str,
    metric_pattern: str = 'deterministic_loss_per_step',
) -> List[LossExtraction]:
    """Parse a SuperBench results-summary.jsonl into per-(node, rank) participants.

    Each line is one node's summary and carries a ``node`` field. Per-step data
    is stored under keys containing ``metric_pattern``; distributed runs emit one
    key per rank (``..._rank0``, ``..._rank1``, ...). Every matching key becomes
    its own participant so per-GPU attribution is preserved.

    Args:
        results_file: Path to the jsonl results file.
        metric_pattern: Substring identifying the per-step metric keys.

    Returns:
        List of LossExtraction objects, one per (node, rank).

    Raises:
        FileNotFoundError: If results_file does not exist.
    """
    path = Path(results_file)
    if not path.exists():
        raise FileNotFoundError(f'Results file not found: {results_file}')

    extractions: List[LossExtraction] = []

    with open(path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f'Skipping malformed JSON at line {line_num}')
                continue

            node_name = record.get('node', f'unknown_node_{line_num}')

            # Collect every key that carries per-step data for this metric. Each
            # rank-suffixed key becomes its own participant.
            matched_any = False
            for key, val in record.items():
                if metric_pattern not in key:
                    continue
                per_step_data = _parse_per_step_value(val)
                if per_step_data is None:
                    logger.warning(f'Cannot parse per-step data for {node_name} at key {key}')
                    continue

                rank_match = _RANK_SUFFIX_RE.search(key)
                rank = int(rank_match.group(1)) if rank_match else None

                losses: Dict[int, Any] = {}
                anomaly_steps: List[int] = []
                for step_str, value in per_step_data.items():
                    _coerce_step_value(step_str, value, losses, anomaly_steps)

                extractions.append(
                    LossExtraction(node_name=node_name, losses=losses, anomaly_steps=anomaly_steps, rank=rank)
                )
                matched_any = True

            if not matched_any:
                logger.info(f'No per-step data for node {node_name} (metric: {metric_pattern}), skipping')

    return extractions


def merge_results_files(
    results_files: Sequence[str], output_file: str, node_names: Optional[Sequence[str]] = None
) -> str:
    """Merge several single-node results-summary.jsonl files into one, labelling nodes.

    A single node's results file is one line -- not enough for a quorum. This
    helper stamps each source file with a distinct ``node`` name and concatenates
    them so the result can be fed to :func:`run_sdc_check`.

    Args:
        results_files: Paths to per-node results-summary.jsonl files.
        output_file: Destination merged jsonl path.
        node_names: Optional explicit node names (defaults to node_0, node_1, ...).

    Returns:
        The output_file path.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If node_names is given but length doesn't match results_files.
    """
    if node_names is not None and len(node_names) != len(results_files):
        raise ValueError(
            f'node_names length ({len(node_names)}) must match results_files length ({len(results_files)}).'
        )

    merged: List[Dict[str, Any]] = []
    for idx, rf in enumerate(results_files):
        p = Path(rf)
        if not p.exists():
            raise FileNotFoundError(f'Results file not found: {rf}')
        node = node_names[idx] if node_names is not None else f'node_{idx}'
        with open(p, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record['node'] = node
                merged.append(record)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for record in merged:
            f.write(json.dumps(record) + '\n')

    return str(out)


def format_quorum_report(result: QuorumResult) -> str:
    """Format a human-readable text report of the quorum result."""
    lines = []
    lines.append('=' * 60)
    lines.append('SDC QUORUM COMPARISON REPORT')
    lines.append('=' * 60)
    lines.append(f'Participants compared: {result.total_participants}')
    lines.append(f'Rank groups voted:     {result.rank_count}')
    lines.append(f'Total steps compared:  {result.total_steps}')
    lines.append(f'Ambiguous steps (ties): {len(result.ambiguous_steps)}')
    lines.append('')

    if not result.has_outliers:
        lines.append('VERDICT: PASS')
        lines.append('All participants produced bit-identical values at every step.')
    else:
        lines.append('VERDICT: FAIL -- SDC DETECTED')
        lines.append(f'Outlier participants: {len(result.outlier_summary)}')
        lines.append('')
        for ident, steps in sorted(result.outlier_summary.items()):
            detail = result.divergence_detail.get(ident, {})
            preview = steps[:20]
            suffix = '...' if len(steps) > 20 else ''
            lines.append(f'  GPU/participant: {ident}')
            lines.append(f'    Outlier at {len(steps)} step(s): {preview}{suffix}')
            lines.append(f'    First divergence at step {detail.get("first_outlier_step")}:')
            lines.append(f'      Majority value: {detail.get("majority_value_at_first_step")}')
            lines.append(f'      This value:     {detail.get("first_outlier_value")}')
            lines.append('')

    lines.append('=' * 60)
    return '\n'.join(lines)


def format_quorum_json(result: QuorumResult) -> Dict[str, Any]:
    """Format the quorum result as a JSON-serializable dict."""
    output: Dict[str, Any] = {
        'verdict': 'FAIL' if result.has_outliers else 'PASS',
        'total_participants': result.total_participants,
        'rank_count': result.rank_count,
        'total_steps': result.total_steps,
        'ambiguous_step_count': len(result.ambiguous_steps),
        'outlier_participant_count': len(result.outlier_summary),
        'outlier_participants': {},
    }

    for ident, steps in result.outlier_summary.items():
        detail = result.divergence_detail.get(ident, {})
        output['outlier_participants'][ident] = {
            'outlier_step_count': len(steps),
            'first_outlier_step': detail.get('first_outlier_step'),
            'first_outlier_value': detail.get('first_outlier_value'),
            'majority_value_at_first_step': detail.get('majority_value_at_first_step'),
            'all_outlier_steps': steps,
        }

    return output


def run_sdc_check(
    raw_data_file: str,
    output_dir: Optional[str] = None,
    output_format: str = 'text',
    metric_pattern: str = 'deterministic_loss_per_step',
) -> QuorumResult:
    """End-to-end SDC quorum check: extract, compare (per rank), report.

    Args:
        raw_data_file: Path to results-summary.jsonl with multi-node data.
        output_dir: Optional directory to write report files.
        output_format: 'text', 'json', or 'both'.
        metric_pattern: Metric substring containing per-step data.

    Returns:
        QuorumResult object.

    Raises:
        FileNotFoundError: If raw_data_file doesn't exist.
        ValueError: If no rank-group has at least 2 participants to compare.
    """
    extractions = extract_from_results(raw_data_file, metric_pattern)

    # Need at least one rank-group with >= 2 participants to hold a vote.
    by_rank: Dict[Optional[int], int] = defaultdict(int)
    for ext in extractions:
        by_rank[ext.rank] += 1
    if not extractions or max(by_rank.values()) < 2:
        raise ValueError(
            f'Need at least 2 comparable participants (same rank across nodes) for quorum '
            f'comparison, found {len(extractions)} participant(s) across {len(by_rank)} rank-group(s). '
            f'Check that the results file contains the metric "{metric_pattern}" for multiple nodes. '
            f'Use merge_results_files() to combine per-node result files first.'
        )

    logger.info(
        f'Running quorum on {len(extractions)} participant(s) across {len(by_rank)} rank-group(s), '
        f'steps range: {min(min(e.losses.keys()) for e in extractions if e.losses)} - '
        f'{max(max(e.losses.keys()) for e in extractions if e.losses)}'
    )

    result = compute_quorum_by_rank(extractions)

    text_report = format_quorum_report(result)
    json_report = format_quorum_json(result)

    print(text_report)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if output_format in ('text', 'both'):
            (out_path / 'sdc_quorum_report.txt').write_text(text_report)
        if output_format in ('json', 'both'):
            (out_path / 'sdc_quorum_report.json').write_text(json.dumps(json_report, indent=2))

    return result
