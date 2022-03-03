# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DataDiagnosis module."""

import json
import unittest
import yaml
from pathlib import Path

import pandas as pd

from superbench.analyzer import DataDiagnosis
import superbench.analyzer.file_handler as file_handler


class TestDataDiagnosis(unittest.TestCase):
    """Test for DataDiagnosis class."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        self.parent_path = Path(__file__).parent
        self.output_excel_file = str(self.parent_path / 'diagnosis_summary.xlsx')
        self.test_rule_file_fake = str(self.parent_path / 'test_rules_fake.yaml')
        self.output_json_file = str(self.parent_path / 'diagnosis_summary.jsonl')

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        for file in [self.output_excel_file, self.output_json_file, self.test_rule_file_fake]:
            p = Path(file)
            if p.is_file():
                p.unlink()

    def test_data_diagnosis(self):
        """Test for rule-based data diagnosis."""
        # Test - read_raw_data and get_metrics_from_raw_data
        # Positive case
        test_raw_data = str(self.parent_path / 'test_results.jsonl')
        test_rule_file = str(self.parent_path / 'test_rules.yaml')
        test_baseline_file = str(self.parent_path / 'test_baseline.json')
        diag1 = DataDiagnosis()
        diag1._raw_data_df = file_handler.read_raw_data(test_raw_data)
        diag1._benchmark_metrics_dict = diag1._get_metrics_by_benchmarks(list(diag1._raw_data_df))
        assert (len(diag1._raw_data_df) == 3)
        # Negative case
        test_raw_data_fake = str(self.parent_path / 'test_results_fake.jsonl')
        test_rule_file_fake = str(self.parent_path / 'test_rules_fake.yaml')
        diag2 = DataDiagnosis()
        diag2._raw_data_df = file_handler.read_raw_data(test_raw_data_fake)
        diag2._benchmark_metrics_dict = diag2._get_metrics_by_benchmarks(list(diag2._raw_data_df))
        assert (len(diag2._raw_data_df) == 0)
        assert (len(diag2._benchmark_metrics_dict) == 0)
        metric_list = [
            'gpu_temperature', 'gpu_power_limit', 'gemm-flops/FP64',
            'bert_models/pytorch-bert-base/steptime_train_float32'
        ]
        self.assertDictEqual(
            diag2._get_metrics_by_benchmarks(metric_list), {
                'gemm-flops': {'gemm-flops/FP64'},
                'bert_models': {'bert_models/pytorch-bert-base/steptime_train_float32'}
            }
        )
        # Test - read rules
        rules = file_handler.read_rules(test_rule_file_fake)
        assert (not rules)
        rules = file_handler.read_rules(test_rule_file)
        assert (rules)
        # Test - _check_and_format_rules
        # Negative case
        false_rules = [
            {
                'criteria': 'lambda x:x>0',
                'categories': 'KernelLaunch',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'criteria': 'lambda x:x>0',
                'function': 'variance',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'categories': 'KernelLaunch',
                'function': 'variance',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'criteria': 'lambda x:x>0',
                'function': 'abb',
                'categories': 'KernelLaunch',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'criteria': 'lambda x:x>0',
                'function': 'abb',
                'categories': 'KernelLaunch',
            }, {
                'criteria': 'x>5',
                'function': 'abb',
                'categories': 'KernelLaunch',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }
        ]
        metric = 'kernel-launch/event_overhead:0'
        for rules in false_rules:
            self.assertRaises(Exception, diag1._check_and_format_rules, rules, metric)
        # Positive case
        true_rules = [
            {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0.05',
                'function': 'variance',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x<-0.05',
                'function': 'variance',
                'metrics': 'kernel-launch/event_overhead:\\d+'
            }, {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0',
                'function': 'value',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }
        ]
        for rules in true_rules:
            assert (diag1._check_and_format_rules(rules, metric))
        # Test - _get_baseline_of_metric
        baseline = file_handler.read_baseline(test_baseline_file)
        assert (diag1._get_baseline_of_metric(baseline, 'kernel-launch/event_overhead:0') == 0.00596)
        assert (diag1._get_baseline_of_metric(baseline, 'kernel-launch/return_code') == 0)
        assert (diag1._get_baseline_of_metric(baseline, 'mem-bw/H2D:0') == -1)
        # Test - _parse_rules_and_baseline
        # Negative case
        fake_rules = file_handler.read_rules(test_rule_file_fake)
        baseline = file_handler.read_baseline(test_baseline_file)
        assert (diag2._parse_rules_and_baseline(fake_rules, baseline) is False)
        diag2 = DataDiagnosis()
        diag2._raw_data_df = file_handler.read_raw_data(test_raw_data)
        diag2._benchmark_metrics_dict = diag2._get_metrics_by_benchmarks(list(diag2._raw_data_df))
        p = Path(test_rule_file)
        with p.open() as f:
            rules = yaml.load(f, Loader=yaml.SafeLoader)
        rules['superbench']['rules']['fake'] = false_rules[0]
        with open(test_rule_file_fake, 'w') as f:
            yaml.dump(rules, f)
        assert (diag1._parse_rules_and_baseline(fake_rules, baseline) is False)
        # Positive case
        rules = file_handler.read_rules(test_rule_file)
        assert (diag1._parse_rules_and_baseline(rules, baseline))
        # Test - _run_diagnosis_rules_for_single_node
        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-01')
        assert (details_row)
        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-02')
        assert (not details_row)
        # Test - _run_diagnosis_rules
        baseline = file_handler.read_baseline(test_baseline_file)
        data_not_accept_df, label_df = diag1.run_diagnosis_rules(rules, baseline)
        assert (len(label_df) == 3)
        assert (label_df.loc['sb-validation-01']['label'] == 1)
        assert (label_df.loc['sb-validation-02']['label'] == 0)
        assert (label_df.loc['sb-validation-03']['label'] == 1)
        node = 'sb-validation-01'
        row = data_not_accept_df.loc[node]
        assert (len(row) == 36)
        assert (row['Category'] == 'KernelLaunch')
        assert (
            row['Defective Details'] ==
            'kernel-launch/event_overhead:0(B/L: 0.0060 VAL: 0.1000 VAR: 1577.85% Rule:lambda x:x>0.05)'
        )
        node = 'sb-validation-03'
        row = data_not_accept_df.loc[node]
        assert (len(row) == 36)
        assert ('FailedTest' in row['Category'])
        assert ('mem-bw/return_code(VAL: 1.0000 Rule:lambda x:x>0)' in row['Defective Details'])
        assert ('mem-bw/H2D_Mem_BW:0_miss' in row['Defective Details'])
        assert (len(data_not_accept_df) == 2)
        # Test - output in excel
        file_handler.output_excel(diag1._raw_data_df, data_not_accept_df, self.output_excel_file, diag1._sb_rules)
        excel_file = pd.ExcelFile(self.output_excel_file, engine='openpyxl')
        data_sheet_name = 'Raw Data'
        raw_data_df = excel_file.parse(data_sheet_name)
        assert (len(raw_data_df) == 3)
        data_sheet_name = 'Not Accept'
        data_not_accept_read_from_excel = excel_file.parse(data_sheet_name)
        assert (len(data_not_accept_read_from_excel) == 2)
        assert ('Category' in data_not_accept_read_from_excel)
        assert ('Defective Details' in data_not_accept_read_from_excel)
        # Test - output in json
        file_handler.output_json_data_not_accept(data_not_accept_df, self.output_json_file)
        assert (Path(self.output_json_file).is_file())
        with Path(self.output_json_file).open() as f:
            data_not_accept_read_from_json = f.readlines()
        assert (len(data_not_accept_read_from_json) == 2)
        for line in data_not_accept_read_from_json:
            json.loads(line)
            assert ('Category' in line)
            assert ('Defective Details' in line)
            assert ('Index' in line)

    def test_data_diagnosis_run(self):
        """Test for the run process of rule-based data diagnosis."""
        test_raw_data = str(self.parent_path / 'test_results.jsonl')
        test_rule_file = str(self.parent_path / 'test_rules.yaml')
        test_baseline_file = str(self.parent_path / 'test_baseline.json')

        # Test - output in excel
        DataDiagnosis().run(test_raw_data, test_rule_file, test_baseline_file, str(self.parent_path), 'excel')
        excel_file = pd.ExcelFile(self.output_excel_file, engine='openpyxl')
        data_sheet_name = 'Not Accept'
        data_not_accept_read_from_excel = excel_file.parse(data_sheet_name)
        expect_result_file = pd.ExcelFile(str(self.parent_path / '../data/diagnosis_summary.xlsx'), engine='openpyxl')
        expect_result = expect_result_file.parse(data_sheet_name)
        pd.testing.assert_frame_equal(data_not_accept_read_from_excel, expect_result)
        # Test - output in json
        DataDiagnosis().run(test_raw_data, test_rule_file, test_baseline_file, str(self.parent_path), 'json')
        assert (Path(self.output_json_file).is_file())
        with Path(self.output_json_file).open() as f:
            data_not_accept_read_from_json = f.read()
        expect_result_file = self.parent_path / '../data/diagnosis_summary.jsonl'
        with Path(expect_result_file).open() as f:
            expect_result = f.read()
        assert (data_not_accept_read_from_json == expect_result)

    def test_mutli_rules(self):
        """Test multi rules check feature."""
        diag1 = DataDiagnosis()
        # test _check_and_format_rules
        false_rules = [
            {
                'criteria': 'lambda x:x>0',
                'categories': 'KernelLaunch',
                'store': 'true',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }
        ]
        metric = 'kernel-launch/event_overhead:0'
        for rules in false_rules:
            self.assertRaises(Exception, diag1._check_and_format_rules, rules, metric)
        # Positive case
        true_rules = [
            {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0.05',
                'store': True,
                'function': 'variance',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'categories': 'CNN',
                'function': 'multi_rules',
                'criteria': 'lambda label:True if label["rule1"]+label["rule2"]>=2 else False'
            }
        ]
        for rules in true_rules:
            assert (diag1._check_and_format_rules(rules, metric))
        # test _run_diagnosis_rules_for_single_node
        rules = {
            'superbench': {
                'rules': {
                    'rule1': {
                        'categories': 'CNN',
                        'criteria': 'lambda x:x<-0.5',
                        'store': True,
                        'function': 'variance',
                        'metrics': ['mem-bw/D2H_Mem_BW']
                    },
                    'rule2': {
                        'categories': 'CNN',
                        'criteria': 'lambda x:x<-0.5',
                        'function': 'variance',
                        'store': True,
                        'metrics': ['kernel-launch/wall_overhead']
                    },
                    'rule3': {
                        'categories': 'CNN',
                        'function': 'multi_rules',
                        'criteria': 'lambda label:True if label["rule1"]+label["rule2"]>=2 else False'
                    }
                }
            }
        }
        baseline = {
            'kernel-launch/wall_overhead': 0.01026,
            'mem-bw/D2H_Mem_BW': 24.3,
        }

        data = {'kernel-launch/wall_overhead': [0.005, 0.005], 'mem-bw/D2H_Mem_BW': [25, 10]}
        diag1._raw_data_df = pd.DataFrame(data, index=['sb-validation-04', 'sb-validation-05'])
        diag1._benchmark_metrics_dict = diag1._get_metrics_by_benchmarks(list(diag1._raw_data_df.columns))
        diag1._parse_rules_and_baseline(rules, baseline)
        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-04')
        assert (not details_row)
        (details_row, summary_data_row) = diag1._run_diagnosis_rules_for_single_node('sb-validation-05')
        assert (details_row)
        assert ('CNN' in details_row[0])
        assert (
            details_row[1] == 'kernel-launch/wall_overhead(B/L: 0.0103 VAL: 0.0050 VAR: -51.27% Rule:lambda x:x<-0.5),'
            + 'mem-bw/D2H_Mem_BW(B/L: 24.3000 VAL: 10.0000 VAR: -58.85% Rule:lambda x:x<-0.5),' +
            'rule3:lambda label:True if label["rule1"]+label["rule2"]>=2 else False'
        )
