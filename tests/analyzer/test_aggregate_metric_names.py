# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for aggregation keys shared by summaries and baseline restoration."""

import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from superbench.analyzer import BaselineGeneration
from superbench.analyzer import data_analysis


@pytest.mark.parametrize(
    'pattern,metrics,expected_key',
    [
        (r'net/size64_gpu(\d+)', ['net/size64_gpu12', 'net/size64_gpu123'], 'net/size64_gpu*'),
        (r'net/size(\d+)_gpu(\d+)', ['net/size64_gpu12', 'net/size128_gpu123'], 'net/size*_gpu*'),
        (r'net/size(?P<size>\d+)_gpu(?P<gpu>\d+)',
         ['net/size64_gpu12', 'net/size128_gpu123'], 'net/size*_gpu*'),
        (r'net/size(\d+)_gpu(\d+)', ['net/size4_gpu7', 'net/size8_gpu9'], 'net/size*_gpu*'),
    ]
)
def test_aggregate_to_baseline_key_roundtrip(pattern, metrics, expected_key):
    """Restore the exact produced key for single and multiple variable-length groups."""
    frame = pd.DataFrame({metrics[0]: [10, 11], metrics[1]: [20, 21]})
    produced = data_analysis.aggregate(frame, pattern)
    assert list(produced.columns) == [expected_key]
    with tempfile.TemporaryDirectory() as directory:
        raw_file = Path(directory) / 'results.jsonl'
        rule_file = Path(directory) / 'summary.yaml'
        rows = [{'node': 'node-{}'.format(index), **row} for index, row in enumerate(frame.to_dict('records'))]
        raw_file.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        rule_file.write_text(json.dumps({'superbench': {'rules': {'network': {
            'statistics': 'mean', 'categories': 'NET', 'aggregate': pattern, 'metrics': ['net/.*'],
        }}}}))
        restored = BaselineGeneration().get_aggregate_data(str(raw_file), str(rule_file))
    assert set(restored.columns) == set(metrics)
    assert restored.shape == (4, 2)
    for metric in metrics:
        assert sorted(restored[metric].tolist()) == [10, 11, 20, 21]


def test_rank_aggregation_name_unchanged():
    """Keep the existing default rank aggregation behavior."""
    frame = pd.DataFrame({'net/value:0': [1, 2], 'net/value:1': [3, 4]})
    pd.testing.assert_frame_equal(data_analysis.aggregate(frame), pd.DataFrame({'net/value': [1, 2, 3, 4]}))


def test_unmatched_pattern_name_unchanged():
    """Keep the producer's existing fallback when the regex does not match."""
    frame = pd.DataFrame({'net/value:0': [1, 2]})
    pd.testing.assert_frame_equal(
        data_analysis.aggregate(frame, r'other/(\d+)'), pd.DataFrame({'net/value': [1, 2]})
    )
