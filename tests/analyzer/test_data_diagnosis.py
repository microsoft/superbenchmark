# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DataDiagnosis module."""

from pathlib import Path

from superbench.analyzer import DataDiagnosis


def test_data_analyzer():
    """Test for rule-based data diagnosis."""
    test_raw_data = str(Path(__file__).parent.resolve()) + '/test_results.jsonl'
    test_rule_file = str(Path(__file__).parent.resolve()) + '/test_rules.yaml'
    print(test_raw_data)
    print(test_rule_file)
    analyzer = DataDiagnosis(test_raw_data)
    assert (len(analyzer._raw_data_df) == 3)
    data_not_accept_df = analyzer.rule_based_diagnosis(test_rule_file)
    node = 'sb-validation-01'
    row = data_not_accept_df.loc[node]
    assert (len(row) == 30)
    assert (row['# of Issues'] == 1)
    assert (row['Category'] == 'kernel-launch')
    assert (row['Issue Details'] == 'kernel-launch/event_overhead:0')
    node = 'sb-validation-03'
    row = data_not_accept_df.loc[node]
    assert (len(row) == 30)
    assert (row['# of Issues'] == 9)
    assert ('mem-bw' in row['Category'])
    assert ('MissTest' in row['Category'])
    assert (len(data_not_accept_df) == 2)
