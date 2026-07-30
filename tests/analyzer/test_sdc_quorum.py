# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for SDC quorum voting module."""

import json
import tempfile

import pytest

from superbench.analyzer.sdc_quorum import (
    LossExtraction,
    compute_quorum,
    compute_quorum_by_rank,
    extract_from_results,
    format_quorum_json,
    format_quorum_report,
    is_anomaly_value,
    merge_results_files,
    run_sdc_check,
    _anomaly_sentinel,
)


class TestComputeQuorum:
    """Tests for the core quorum voting logic."""

    def test_all_nodes_agree(self):
        """All nodes produce identical values -> PASS, no outliers."""
        extractions = [
            LossExtraction(node_name='node_0', losses={1: 2.345, 2: 2.340, 3: 2.335}),
            LossExtraction(node_name='node_1', losses={1: 2.345, 2: 2.340, 3: 2.335}),
            LossExtraction(node_name='node_2', losses={1: 2.345, 2: 2.340, 3: 2.335}),
        ]
        result = compute_quorum(extractions)

        assert result.total_steps == 3
        assert not result.has_outliers
        assert len(result.outlier_summary) == 0
        assert len(result.ambiguous_steps) == 0
        for step in [1, 2, 3]:
            assert result.steps[step].majority_nodes == ['node_0', 'node_1', 'node_2']
            assert result.steps[step].outlier_nodes == []
            assert not result.steps[step].ambiguous

    def test_single_outlier(self):
        """One node diverges at one step -> that node is outlier."""
        extractions = [
            LossExtraction(node_name='node_0', losses={1: 2.345, 2: 2.340, 3: 2.335}),
            LossExtraction(node_name='node_1', losses={1: 2.345, 2: 2.340, 3: 2.335}),
            LossExtraction(node_name='node_2', losses={1: 2.345, 2: 9.999, 3: 2.335}),
        ]
        result = compute_quorum(extractions)

        assert result.has_outliers
        assert result.outlier_summary['node_2'] == [2]
        assert result.steps[2].outlier_nodes == ['node_2']
        assert result.steps[2].majority_value == 2.340
        assert result.steps[1].outlier_nodes == []
        assert result.steps[3].outlier_nodes == []

    def test_multiple_steps_diverge(self):
        """One node diverges at multiple steps."""
        extractions = [
            LossExtraction(node_name='node_0', losses={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}),
            LossExtraction(node_name='node_1', losses={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}),
            LossExtraction(node_name='node_2', losses={1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}),
            LossExtraction(node_name='bad_node', losses={1: 1.0, 2: 7.7, 3: 8.8, 4: 4.0}),
        ]
        result = compute_quorum(extractions)

        assert result.has_outliers
        assert result.outlier_summary['bad_node'] == [2, 3]
        assert 'node_0' not in result.outlier_summary

    def test_tie_produces_ambiguous(self):
        """Two groups of equal size -> ambiguous, no outliers."""
        extractions = [
            LossExtraction(node_name='node_0', losses={1: 1.0}),
            LossExtraction(node_name='node_1', losses={1: 2.0}),
        ]
        result = compute_quorum(extractions)

        assert not result.has_outliers
        assert result.steps[1].ambiguous
        assert result.steps[1].majority_value is None
        assert 1 in result.ambiguous_steps

    def test_tie_at_top_with_runners_up_is_ambiguous(self):
        """3-3-1 split: leader not unique -> ambiguous even with a runner-up."""
        extractions = [
            LossExtraction(node_name='a0', losses={1: 1.0}),
            LossExtraction(node_name='a1', losses={1: 1.0}),
            LossExtraction(node_name='a2', losses={1: 1.0}),
            LossExtraction(node_name='b0', losses={1: 2.0}),
            LossExtraction(node_name='b1', losses={1: 2.0}),
            LossExtraction(node_name='b2', losses={1: 2.0}),
            LossExtraction(node_name='c0', losses={1: 3.0}),
        ]
        result = compute_quorum(extractions)

        assert not result.has_outliers
        assert result.steps[1].ambiguous
        assert result.steps[1].majority_value is None
        assert result.steps[1].outlier_nodes == []

    def test_plurality_not_majority(self):
        """Largest group wins even if not >50%, as long as uniquely largest."""
        extractions = [
            LossExtraction(node_name='node_0', losses={1: 5.0}),
            LossExtraction(node_name='node_1', losses={1: 5.0}),
            LossExtraction(node_name='node_2', losses={1: 5.0}),
            LossExtraction(node_name='node_3', losses={1: 6.0}),
            LossExtraction(node_name='node_4', losses={1: 7.0}),
        ]
        result = compute_quorum(extractions)

        assert result.steps[1].majority_value == 5.0
        assert set(result.steps[1].majority_nodes) == {'node_0', 'node_1', 'node_2'}
        assert set(result.steps[1].outlier_nodes) == {'node_3', 'node_4'}
        assert not result.steps[1].ambiguous

    def test_plurality_tiny_lead_still_wins(self):
        """2-1-1-1-1-1-1 split: unique leader wins even at 25% support."""
        extractions = [
            LossExtraction(node_name='n0', losses={1: 1.0}),
            LossExtraction(node_name='n1', losses={1: 1.0}),
            LossExtraction(node_name='n2', losses={1: 2.0}),
            LossExtraction(node_name='n3', losses={1: 3.0}),
            LossExtraction(node_name='n4', losses={1: 4.0}),
            LossExtraction(node_name='n5', losses={1: 5.0}),
        ]
        result = compute_quorum(extractions)

        assert not result.steps[1].ambiguous
        assert result.steps[1].majority_value == 1.0
        assert sorted(result.steps[1].majority_nodes) == ['n0', 'n1']
        assert len(result.steps[1].outlier_nodes) == 4

    def test_first_seen_ordering_preserved(self):
        """Outliers are reported in first-seen order (parity with reference)."""
        extractions = [
            LossExtraction(node_name='n0', losses={1: 1.0}),
            LossExtraction(node_name='n1', losses={1: 1.0}),
            LossExtraction(node_name='n2', losses={1: 2.0}),
            LossExtraction(node_name='n3', losses={1: 3.0}),
        ]
        result = compute_quorum(extractions)
        assert result.steps[1].outlier_nodes == ['n2', 'n3']

    def test_missing_steps_handled(self):
        """Nodes with missing steps listed in missing_nodes."""
        extractions = [
            LossExtraction(node_name='node_0', losses={1: 1.0, 2: 2.0, 3: 3.0}),
            LossExtraction(node_name='node_1', losses={1: 1.0, 2: 2.0}),
            LossExtraction(node_name='node_2', losses={1: 1.0, 2: 2.0, 3: 3.0}),
        ]
        result = compute_quorum(extractions)

        assert result.steps[3].missing_nodes == ['node_1']
        assert result.steps[3].majority_value == 3.0

    def test_nan_sentinel_is_outlier(self):
        """A NaN sentinel value never matches a real float -> automatic outlier."""
        extractions = [
            LossExtraction(node_name='node_0', losses={1: 2.5, 2: 2.4}),
            LossExtraction(node_name='node_1', losses={1: 2.5, 2: 2.4}),
            LossExtraction(node_name='bad', losses={1: 2.5, 2: _anomaly_sentinel('nan')}),
        ]
        result = compute_quorum(extractions)

        assert result.has_outliers
        assert result.outlier_summary['bad'] == [2]

    def test_empty_extractions(self):
        """Empty list produces empty result."""
        result = compute_quorum([])
        assert result.total_steps == 0
        assert not result.has_outliers

    def test_bit_exact_comparison(self):
        """Even a 1-ULP floating-point difference is caught."""
        import struct

        val_a = 2.345678901234567
        packed = struct.pack('d', val_a)
        int_val = int.from_bytes(packed, 'little') + 1
        val_b = struct.unpack('d', int_val.to_bytes(8, 'little'))[0]

        assert val_a != val_b

        extractions = [
            LossExtraction(node_name='node_0', losses={1: val_a}),
            LossExtraction(node_name='node_1', losses={1: val_a}),
            LossExtraction(node_name='node_2', losses={1: val_b}),
        ]
        result = compute_quorum(extractions)

        assert result.has_outliers
        assert 'node_2' in result.outlier_summary


class TestComputeQuorumByRank:
    """Tests for rank-aware (per-GPU) quorum voting."""

    def test_same_rank_across_nodes_compared(self):
        """Rank k is compared across nodes; a divergent GPU is attributed as node.rankN."""
        extractions = [
            LossExtraction(node_name='nodeA', rank=0, losses={1: 1.0, 2: 2.0}),
            LossExtraction(node_name='nodeB', rank=0, losses={1: 1.0, 2: 2.0}),
            LossExtraction(node_name='nodeC', rank=0, losses={1: 1.0, 2: 9.9}),  # bad GPU
            LossExtraction(node_name='nodeA', rank=1, losses={1: 5.0, 2: 6.0}),
            LossExtraction(node_name='nodeB', rank=1, losses={1: 5.0, 2: 6.0}),
            LossExtraction(node_name='nodeC', rank=1, losses={1: 5.0, 2: 6.0}),
        ]
        result = compute_quorum_by_rank(extractions)

        assert result.has_outliers
        assert result.rank_count == 2
        assert 'nodeC.rank0' in result.outlier_summary
        assert result.outlier_summary['nodeC.rank0'] == [2]
        # rank1 is clean on every node
        assert not any(k.endswith('rank1') for k in result.outlier_summary)

    def test_different_ranks_not_cross_compared(self):
        """Different ranks (different data shards) are never compared to each other."""
        extractions = [
            LossExtraction(node_name='nodeA', rank=0, losses={1: 1.0}),
            LossExtraction(node_name='nodeB', rank=0, losses={1: 1.0}),
            LossExtraction(node_name='nodeA', rank=1, losses={1: 99.0}),  # legitimately different shard
            LossExtraction(node_name='nodeB', rank=1, losses={1: 99.0}),
        ]
        result = compute_quorum_by_rank(extractions)
        assert not result.has_outliers

    def test_single_rank_none_degenerates_to_plain(self):
        """No rank tags -> behaves like plain compute_quorum with node identities."""
        extractions = [
            LossExtraction(node_name='n0', losses={1: 1.0}),
            LossExtraction(node_name='n1', losses={1: 1.0}),
            LossExtraction(node_name='n2', losses={1: 2.0}),
        ]
        result = compute_quorum_by_rank(extractions)
        assert result.outlier_summary['n2'] == [1]


class TestExtractFromResults:
    """Tests for parsing results-summary.jsonl."""

    def _write(self, records):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        for record in records:
            f.write(json.dumps(record) + '\n')
        f.flush()
        f.close()
        return f.name

    def test_basic_extraction(self):
        """Parse a simple jsonl file with per-step data (no rank suffix)."""
        path = self._write([
            {'node': 'node_A', 'deterministic_loss_per_step': json.dumps({0: 1.0, 100: 1.5, 200: 2.0})},
            {'node': 'node_B', 'deterministic_loss_per_step': json.dumps({0: 1.0, 100: 1.5, 200: 2.0})},
        ])
        extractions = extract_from_results(path)

        assert len(extractions) == 2
        assert extractions[0].node_name == 'node_A'
        assert extractions[0].rank is None
        assert extractions[0].losses == {0: 1.0, 100: 1.5, 200: 2.0}

    def test_per_rank_keys_expand_to_participants(self):
        """Distributed record with multiple _rankN keys -> one participant per rank."""
        path = self._write([
            {
                'node': 'node_A',
                'model/deterministic_loss_per_step_rank0': [json.dumps({0: 1.0, 1: 1.5})],
                'model/deterministic_loss_per_step_rank1': [json.dumps({0: 2.0, 1: 2.5})],
            },
        ])
        extractions = extract_from_results(path)

        assert len(extractions) == 2
        ranks = sorted(e.rank for e in extractions)
        assert ranks == [0, 1]
        by_rank = {e.rank: e for e in extractions}
        assert by_rank[0].losses == {0: 1.0, 1: 1.5}
        assert by_rank[1].identity == 'node_A.rank1'

    def test_nan_becomes_sentinel(self):
        """None values become anomaly sentinels."""
        path = self._write([
            {'node': 'node_A', 'deterministic_loss_per_step': json.dumps({0: 1.0, 100: None, 200: 2.0})},
        ])
        extractions = extract_from_results(path)

        assert is_anomaly_value(extractions[0].losses[100])
        assert 100 in extractions[0].anomaly_steps

    def test_missing_metric_skips_node(self):
        """Nodes without per-step metric are skipped."""
        path = self._write([
            {'node': 'node_A', 'deterministic_loss_per_step': json.dumps({0: 1.0})},
            {'node': 'node_B', 'some_other_metric': 42},
        ])
        extractions = extract_from_results(path)

        assert len(extractions) == 1
        assert extractions[0].node_name == 'node_A'

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_from_results('/nonexistent/path.jsonl')


class TestMergeResultsFiles:
    """Tests for the multi-node merge helper."""

    def _write_one(self, record):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        f.write(json.dumps(record) + '\n')
        f.flush()
        f.close()
        return f.name

    def test_merge_labels_nodes(self):
        f0 = self._write_one({'deterministic_loss_per_step': json.dumps({0: 1.0})})
        f1 = self._write_one({'deterministic_loss_per_step': json.dumps({0: 1.0})})
        out = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name

        merge_results_files([f0, f1], out)
        with open(out) as f:
            records = [json.loads(line) for line in f]

        assert [r['node'] for r in records] == ['node_0', 'node_1']

    def test_merge_custom_names(self):
        f0 = self._write_one({'x': 1})
        f1 = self._write_one({'x': 2})
        out = tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False).name
        merge_results_files([f0, f1], out, node_names=['hostA', 'hostB'])
        with open(out) as f:
            nodes = [json.loads(line)['node'] for line in f]
        assert nodes == ['hostA', 'hostB']

    def test_merge_name_length_mismatch(self):
        f0 = self._write_one({'x': 1})
        with pytest.raises(ValueError, match='must match'):
            merge_results_files([f0], 'out.jsonl', node_names=['a', 'b'])


class TestFormatters:
    """Tests for report formatting."""

    def test_pass_report(self):
        extractions = [
            LossExtraction(node_name='n0', losses={1: 1.0}),
            LossExtraction(node_name='n1', losses={1: 1.0}),
        ]
        result = compute_quorum(extractions)
        report = format_quorum_report(result)
        assert 'PASS' in report
        assert 'bit-identical' in report

    def test_fail_report(self):
        extractions = [
            LossExtraction(node_name='good_0', losses={1: 1.0}),
            LossExtraction(node_name='good_1', losses={1: 1.0}),
            LossExtraction(node_name='bad_node', losses={1: 9.9}),
        ]
        result = compute_quorum(extractions)
        report = format_quorum_report(result)
        assert 'FAIL' in report
        assert 'bad_node' in report

    def test_json_format(self):
        extractions = [
            LossExtraction(node_name='n0', losses={1: 1.0, 2: 2.0}),
            LossExtraction(node_name='n1', losses={1: 1.0, 2: 2.0}),
            LossExtraction(node_name='n2', losses={1: 1.0, 2: 5.0}),
        ]
        result = compute_quorum(extractions)
        output = format_quorum_json(result)

        assert output['verdict'] == 'FAIL'
        assert output['total_steps'] == 2
        assert 'n2' in output['outlier_participants']
        assert output['outlier_participants']['n2']['outlier_step_count'] == 1
        assert output['outlier_participants']['n2']['first_outlier_step'] == 2


class TestRunSdcCheck:
    """Integration tests for run_sdc_check."""

    def _write(self, records):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        for record in records:
            f.write(json.dumps(record) + '\n')
        f.flush()
        f.close()
        return f.name

    def test_end_to_end_pass(self):
        path = self._write([
            {'node': 'node_0', 'deterministic_loss_per_step': json.dumps({0: 1.0, 1: 1.5, 2: 2.0})},
            {'node': 'node_1', 'deterministic_loss_per_step': json.dumps({0: 1.0, 1: 1.5, 2: 2.0})},
            {'node': 'node_2', 'deterministic_loss_per_step': json.dumps({0: 1.0, 1: 1.5, 2: 2.0})},
        ])
        result = run_sdc_check(path, output_format='json')
        assert not result.has_outliers

    def test_end_to_end_fail(self):
        path = self._write([
            {'node': 'node_0', 'deterministic_loss_per_step': json.dumps({0: 1.0, 1: 1.5, 2: 2.0})},
            {'node': 'node_1', 'deterministic_loss_per_step': json.dumps({0: 1.0, 1: 1.5, 2: 2.0})},
            {'node': 'bad_node', 'deterministic_loss_per_step': json.dumps({0: 1.0, 1: 9.9, 2: 2.0})},
        ])
        result = run_sdc_check(path, output_format='json')
        assert result.has_outliers
        assert 'bad_node' in result.outlier_summary

    def test_per_rank_fail_attributes_gpu(self):
        """Three nodes, 2 GPUs each; one GPU corrupts a step -> attributed as node.rankN.

        Per-rank grouping compares the same global_rank across nodes. Attribution
        needs at least three participants per rank-group so the divergent one is a
        minority against an unambiguous plurality (two nodes vs one is 2-1, not a
        1-1 tie).
        """
        path = self._write([
            {
                'node': 'node_0',
                'm/deterministic_loss_per_step_rank0': json.dumps({0: 1.0, 1: 1.5}),
                'm/deterministic_loss_per_step_rank1': json.dumps({0: 3.0, 1: 3.5}),
            },
            {
                'node': 'node_1',
                'm/deterministic_loss_per_step_rank0': json.dumps({0: 1.0, 1: 1.5}),
                'm/deterministic_loss_per_step_rank1': json.dumps({0: 3.0, 1: 9.9}),  # bad GPU
            },
            {
                'node': 'node_2',
                'm/deterministic_loss_per_step_rank0': json.dumps({0: 1.0, 1: 1.5}),
                'm/deterministic_loss_per_step_rank1': json.dumps({0: 3.0, 1: 3.5}),
            },
        ])
        result = run_sdc_check(path, output_format='json')
        assert result.has_outliers
        assert 'node_1.rank1' in result.outlier_summary
        assert result.outlier_summary['node_1.rank1'] == [1]
        # The healthy rank1 peers and all rank0 participants must not be outliers.
        assert 'node_0.rank1' not in result.outlier_summary
        assert 'node_2.rank1' not in result.outlier_summary

    def test_insufficient_participants_raises(self):
        path = self._write([
            {'node': 'only_node', 'deterministic_loss_per_step': json.dumps({0: 1.0})},
        ])
        with pytest.raises(ValueError, match='at least 2'):
            run_sdc_check(path)
