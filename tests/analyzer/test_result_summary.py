# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ResultSummary module."""

import unittest
import yaml
from pathlib import Path

import pandas as pd

from superbench.analyzer import ResultSummary
import superbench.analyzer.file_handler as file_handler


class TestResultSummary(unittest.TestCase):
    """Test for ResultSummary class."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        self.parent_path = Path(__file__).parent
        self.output_excel_file = str(self.parent_path / 'results-summary.xlsx')
        self.output_md_file = str(self.parent_path / 'results-summary.md')
        self.output_html_file = str(self.parent_path / 'results-summary.html')
        self.test_rule_file_fake = str(self.parent_path / 'test_rules_fake.yaml')
        self.test_raw_data = str(self.parent_path / 'test_results.jsonl')
        self.test_rule_file = str(self.parent_path / 'test_summary_rules.yaml')

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        for file in [self.output_excel_file, self.test_rule_file_fake, self.output_md_file, self.output_html_file]:
            p = Path(file)
            if p.is_file():
                p.unlink()

    def test_result_summary(self):
        """Test result summary class."""
        rs1 = ResultSummary()
        rs1._raw_data_df = file_handler.read_raw_data(self.test_raw_data)
        rs1._benchmark_metrics_dict = rs1._get_metrics_by_benchmarks(list(rs1._raw_data_df))
        # Test - _check_rules
        # Negative case
        false_rules = [
            {
                'categories': 'KernelLaunch',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'categories': 'KernelLaunch',
                'statistics': 'abb',
                'metrics': ['kernel-launch/event_overhead:\\d+']
            }, {
                'categories': 'KernelLaunch',
                'statistics': 'mean',
                'metrics': ['kernel-launch/event_overhead:\\d+'],
                'aggregate': 'abb'
            }
        ]
        metric = 'kernel-launch/event_overhead:0'
        for rules in false_rules:
            self.assertRaises(Exception, rs1._check_rules, rules, metric)
        # Positive case
        true_rules = [
            {
                'categories': 'KernelLaunch',
                'statistics': 'mean',
                'metrics': ['kernel-launch/event_overhead:\\d+'],
                'aggregate': True
            },
            {
                'categories': 'KernelLaunch',
                'statistics': ['mean', 'p50'],
                'metrics': ['kernel-launch/event_overhead:\\d+']
            },
            {
                'categories': 'KernelLaunch',
                'statistics': 'mean',
                'metrics': ['kernel-launch/event_overhead:\\d+'],
                'aggregate': 'kernel-launch/event_overhead(:\\d+)'
            },
        ]
        for rules in true_rules:
            assert (rs1._check_rules(rules, metric))

        # Test - _parse_rules
        # Negative case
        rs2 = ResultSummary()
        self.assertRaises(Exception, file_handler.read_rules, self.test_rule_file_fake)
        rs2._raw_data_df = file_handler.read_raw_data(self.test_raw_data)
        rs2._benchmark_metrics_dict = rs2._get_metrics_by_benchmarks(list(rs2._raw_data_df))
        p = Path(self.test_rule_file)
        with p.open() as f:
            rules = yaml.load(f, Loader=yaml.SafeLoader)
        rules['superbench']['rules']['fake'] = false_rules[0]
        with open(self.test_rule_file_fake, 'w') as f:
            yaml.dump(rules, f)
        assert (rs1._parse_rules([]) is False)
        # Positive case
        rules = file_handler.read_rules(self.test_rule_file)
        assert (rs1._parse_rules(rules))

        # Test - _generate_summary
        summary = rs1._generate_summary(round=2)
        assert (len(summary) == 3)

        # Test - _merge_summary
        expected_summary_merge = [
            ['KernelLaunch', 'kernel-launch/event_overhead', 'mean', 0.0097],
            ['KernelLaunch', 'kernel-launch/event_overhead', 'p90', 0.006],
            ['KernelLaunch', 'kernel-launch/event_overhead', 'min', 0.0055],
            ['KernelLaunch', 'kernel-launch/event_overhead', 'max', 0.1],
            ['KernelLaunch', 'kernel-launch/wall_overhead', 'mean', 0.01],
            ['KernelLaunch', 'kernel-launch/wall_overhead', 'p90', 0.011],
            ['KernelLaunch', 'kernel-launch/wall_overhead', 'min', 0.01],
            ['KernelLaunch', 'kernel-launch/wall_overhead', 'max', 0.011],
            ['NCCL', 'nccl-bw/allreduce_8388608_busbw:0', 'mean', 89.51],
            ['RDMA', 'ib-loopback/IB_write_8388608_Avg_*:0', 'mean', 23925.84]
        ]
        expected_summary_merge_df = pd.DataFrame(expected_summary_merge)
        summary_merge_df = rs1._merge_summary(summary)
        pd.testing.assert_frame_equal(expected_summary_merge_df, summary_merge_df)

    def test_no_matched_rule(self):
        """Test for support no matching rules."""
        # Positive case
        rules = {
            'superbench': {
                'rules': {
                    'fake': {
                        'categories': 'FAKE',
                        'statistics': ['mean', 'max'],
                        'metrics': ['abb/fake:\\d+'],
                        'aggregate': True
                    }
                }
            }
        }
        rs1 = ResultSummary()
        rs1._raw_data_df = file_handler.read_raw_data(self.test_raw_data)
        rs1._benchmark_metrics_dict = rs1._get_metrics_by_benchmarks(list(rs1._raw_data_df))
        assert (rs1._parse_rules(rules))
        summary = rs1._generate_summary(round=2)
        assert (len(summary) == 1)
        assert (summary['FAKE'] == [['FAKE', '', 'mean', ''], ['FAKE', '', 'max', '']])

    def test_result_summary_run(self):
        """Test for the run process of result summary."""
        # Test - output in excel
        ResultSummary().run(self.test_raw_data, self.test_rule_file, str(self.parent_path), 'excel', round=2)
        excel_file = pd.ExcelFile(self.output_excel_file, engine='openpyxl')
        data_sheet_name = 'Summary'
        summary = excel_file.parse(data_sheet_name, header=None)
        expect_result_file = pd.ExcelFile(str(self.parent_path / '../data/results_summary.xlsx'), engine='openpyxl')
        expect_result = expect_result_file.parse(data_sheet_name, header=None)
        pd.testing.assert_frame_equal(summary, expect_result)

        # Test - output in md
        ResultSummary().run(self.test_raw_data, self.test_rule_file, str(self.parent_path), 'md', round=2)
        expected_md_file = str(self.parent_path / '../data/results_summary.md')
        with open(expected_md_file, 'r') as f:
            expect_result = f.read()
        with open(self.output_md_file, 'r') as f:
            summary = f.read()
        assert (summary == expect_result)

        # Test - output in html
        ResultSummary().run(self.test_raw_data, self.test_rule_file, str(self.parent_path), 'html', round=2)
        expected_html_file = str(self.parent_path / '../data/results_summary.html')
        with open(expected_html_file, 'r') as f:
            expect_result = f.read()
        with open(self.output_html_file, 'r') as f:
            summary = f.read()
        assert (summary == expect_result)
