# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DataDiagnosis module."""

import unittest
import yaml
from pathlib import Path

from superbench.analyzer import DataDiagnosis


class TestDataDiagnosis(unittest.TestCase):
    """Test for DataDiagnosis class."""
    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        test_rule_file_fake = str(Path(__file__).parent.resolve()) + '/test_rules_fake.yaml'
        p = Path(test_rule_file_fake)
        if p.is_file():
            p.unlink()

    def test_data_diagnosis(self):
        """Test for rule-based data diagnosis."""
        test_raw_data = str(Path(__file__).parent.resolve()) + '/test_results.jsonl'
        test_rule_file = str(Path(__file__).parent.resolve()) + '/test_rules.yaml'
        diag1 = DataDiagnosis(test_raw_data)
        assert (len(diag1._raw_data_df) == 3)

        test_raw_data_fake = str(Path(__file__).parent.resolve()) + '/test_results_fake.jsonl'
        test_rule_file_fake = str(Path(__file__).parent.resolve()) + '/test_rules_fake.yaml'
        diag2 = DataDiagnosis(test_raw_data_fake)
        diag2._read_raw_data(test_raw_data_fake)
        diag2._get_metrics_from_raw_data()
        assert (len(diag2._raw_data_df) == 0)
        assert (len(diag2._metrics) == 0)

        baseline = diag2._read_baseline(test_rule_file_fake)
        assert (not baseline)
        baseline = diag1._read_baseline(test_rule_file)
        assert (baseline)

        false_baselines = [
            {
                'criteria': 0.05
            }, {
                'criteria': 'a',
                'rules': {
                    'name': 'value'
                }
            }, {
                'criteria': 0.05,
                'rules': {
                    'name': 'fake'
                }
            }, {
                'criteria': 0.05,
                'rules': {
                    'name': 'variance'
                }
            }
        ]
        metric = 'kernel-launch/event_overhead:0'
        for baseline in false_baselines:
            self.assertRaises(Exception, diag1._check_baseline, baseline, metric)
        true_baselines = [
            {
                'criteria': 5,
                'rules': {
                    'name': 'value'
                }
            }, {
                'criteria': 0.05,
                'rules': {
                    'name': 'variance',
                    'condition': -0.05
                }
            }
        ]
        for baseline in true_baselines:
            assert (diag1._check_baseline(baseline, metric))

        assert (diag2._get_criteria(test_rule_file_fake) is False)
        diag2 = DataDiagnosis(test_raw_data)
        p = Path(test_rule_file)
        with p.open() as f:
            baseline = yaml.load(f, Loader=yaml.SafeLoader)
        baseline['superbench']['benchmarks']['kernel-launch']['metrics']['kernel-launch/event_overhead:\\d+'
                                                                         ] = false_baselines[0]
        with open(test_rule_file_fake, 'w') as f:
            yaml.dump(baseline, f)
        assert (diag1._get_criteria(test_rule_file_fake) is False)

        assert (diag1._get_criteria(test_rule_file))

        benchmark_list = {'mem-bw', 'bert_models'}
        assert (diag1.hw_issue(benchmark_list))
        benchmark_list = {'tensorrt-inference', 'bert_models', 'ort-inference'}
        assert (diag1.hw_issue(benchmark_list) is False)

        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-01')
        assert (details_row)
        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-02')
        assert (not details_row)

        data_not_accept_df = diag1.run_diagnosis_rules(test_rule_file)
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
