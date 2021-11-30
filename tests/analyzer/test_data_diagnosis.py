# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DataDiagnosis module."""

import pandas as pd
import unittest
import yaml
from pathlib import Path

from superbench.analyzer import DataDiagnosis
import superbench.analyzer.file_handler as file_handler


class TestDataDiagnosis(unittest.TestCase):
    """Test for DataDiagnosis class."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        self.output_file = str(Path(__file__).parent.resolve()) + '/results_summary.xlsx'
        self.test_rule_file_fake = str(Path(__file__).parent.resolve()) + '/test_rules_fake.yaml'

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        p = Path(self.test_rule_file_fake)
        if p.is_file():
            p.unlink()
        p = Path(self.output_file)
        if p.is_file():
            p.unlink()

    def test_data_diagnosis(self):
        """Test for rule-based data diagnosis."""
        # Test - read_raw_data and get_metrics_from_raw_data
        # Positive case
        test_raw_data = str(Path(__file__).parent.resolve()) + '/test_results.jsonl'
        test_rule_file = str(Path(__file__).parent.resolve()) + '/test_rules.yaml'
        diag1 = DataDiagnosis()
        diag1._raw_data_df = file_handler.read_raw_data(test_raw_data)
        diag1._get_metrics_from_raw_data()
        assert (len(diag1._raw_data_df) == 3)
        # Negative case
        test_raw_data_fake = str(Path(__file__).parent.resolve()) + '/test_results_fake.jsonl'
        test_rule_file_fake = str(Path(__file__).parent.resolve()) + '/test_rules_fake.yaml'
        diag2 = DataDiagnosis()
        diag2._raw_data_df = file_handler.read_raw_data(test_raw_data_fake)
        diag2._get_metrics_from_raw_data()
        assert (len(diag2._raw_data_df) == 0)
        assert (len(diag2._metrics) == 0)
        # Test - read baseline
        baseline = file_handler.read_baseline(test_rule_file_fake)
        assert (not baseline)
        baseline = file_handler.read_baseline(test_rule_file)
        assert (baseline)
        # Test - _check_baseline
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
            }, {
                'rules': {
                    'name': 'variance',
                    'condition': -0.05
                }
            }
        ]
        for baseline in true_baselines:
            assert (diag1._check_baseline(baseline, metric))
        # Test - _get_criteria
        # Negative case
        assert (diag2._get_criteria(test_rule_file_fake) is False)
        diag2 = DataDiagnosis()
        diag2._raw_data_df = file_handler.read_raw_data(test_raw_data)
        diag2._get_metrics_from_raw_data()
        p = Path(test_rule_file)
        with p.open() as f:
            baseline = yaml.load(f, Loader=yaml.SafeLoader)
        baseline['superbench']['benchmarks']['kernel-launch']['metrics']['kernel-launch/event_overhead:\\d+'
                                                                         ] = false_baselines[0]
        with open(test_rule_file_fake, 'w') as f:
            yaml.dump(baseline, f)
        assert (diag1._get_criteria(test_rule_file_fake) is False)
        # Positive case
        assert (diag1._get_criteria(test_rule_file))
        # Test - _run_diagnosis_rules_for_single_node
        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-01')
        assert (details_row)
        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-02')
        assert (not details_row)
        # Test - _run_diagnosis_rules
        data_not_accept_df, label_df = diag1.run_diagnosis_rules(test_rule_file)
        assert (len(label_df) == 3)
        assert (label_df.loc['sb-validation-01']['label'] == 1)
        assert (label_df.loc['sb-validation-02']['label'] == 0)
        assert (label_df.loc['sb-validation-03']['label'] == 1)
        node = 'sb-validation-01'
        row = data_not_accept_df.loc[node]
        assert (len(row) == 29)
        assert (row['# of Issues'] == 1)
        assert (row['Category'] == 'kernel-launch')
        assert (row['Issue Details'] == 'kernel-launch/event_overhead:0(B/L: 0.0063 VAL: 0.1000 VAR: 1491.8816%)')
        node = 'sb-validation-03'
        row = data_not_accept_df.loc[node]
        assert (len(row) == 29)
        assert (row['# of Issues'] == 9)
        assert ('MissTest' in row['Category'])
        assert (len(data_not_accept_df) == 2)
        # Test - excel_output
        diag1.excel_output(data_not_accept_df, str(Path(__file__).parent.resolve()))
        excel_file = pd.ExcelFile(self.output_file)
        data_sheet_name = 'Raw Data'
        raw_data_df = excel_file.parse(data_sheet_name)
        assert (len(raw_data_df) == 3)
        data_sheet_name = 'Not Accept'
        data_not_accept_df = excel_file.parse(data_sheet_name)
        assert (len(data_not_accept_df) == 2)
