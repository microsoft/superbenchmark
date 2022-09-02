# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RuleBase module."""

import unittest
from pathlib import Path

from superbench.analyzer import RuleBase
import superbench.analyzer.file_handler as file_handler


class TestRuleBase(unittest.TestCase):
    """Test for RuleBase class."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        self.parent_path = Path(__file__).parent

    def test_rule_base(self):
        """Test for rule-based functions."""
        # Test - read_raw_data and get_metrics_from_raw_data
        # Positive case
        test_raw_data = str(self.parent_path / 'test_results.jsonl')
        test_rule_file = str(self.parent_path / 'test_rules.yaml')
        rulebase1 = RuleBase()
        rulebase1._raw_data_df = file_handler.read_raw_data(test_raw_data)
        rulebase1._benchmark_metrics_dict = rulebase1._get_metrics_by_benchmarks(list(rulebase1._raw_data_df))
        assert (len(rulebase1._raw_data_df) == 3)
        # Negative case
        test_rule_file_fake = str(self.parent_path / 'test_rules_fake.yaml')

        test_raw_data_fake = str(self.parent_path / 'test_results_fake.jsonl')
        rulebase2 = RuleBase()
        self.assertRaises(FileNotFoundError, file_handler.read_raw_data, test_raw_data_fake)
        rulebase2._benchmark_metrics_dict = rulebase2._get_metrics_by_benchmarks([])
        assert (len(rulebase2._benchmark_metrics_dict) == 0)
        metric_list = [
            'gpu_temperature', 'gpu_power_limit', 'gemm-flops/FP64',
            'bert_models/pytorch-bert-base/steptime_train_float32'
        ]
        self.assertDictEqual(
            rulebase2._get_metrics_by_benchmarks(metric_list), {
                'gemm-flops': {'gemm-flops/FP64'},
                'bert_models': {'bert_models/pytorch-bert-base/steptime_train_float32'}
            }
        )

        # Test - _preprocess
        self.assertRaises(Exception, rulebase1._preprocess, test_raw_data_fake, test_rule_file)
        self.assertRaises(Exception, rulebase1._preprocess, test_raw_data, test_rule_file_fake)
        rules = rulebase1._preprocess(test_raw_data, test_rule_file)
        assert (rules)

        # Test - _check_and_format_rules
        # Negative case
        false_rule = {
            'criteria': 'lambda x:x>0',
            'function': 'variance',
            'metrics': ['kernel-launch/event_overhead:\\d+']
        }
        metric = 'kernel-launch/event_overhead:0'
        self.assertRaises(Exception, rulebase1._check_and_format_rules, false_rule, metric)
        # Positive case
        true_rule = {
            'categories': 'KernelLaunch',
            'criteria': 'lambda x:x<-0.05',
            'function': 'variance',
            'metrics': 'kernel-launch/event_overhead:\\d+'
        }
        true_rule = rulebase1._check_and_format_rules(true_rule, metric)
        assert (true_rule)
        assert (true_rule['metrics'] == ['kernel-launch/event_overhead:\\d+'])

        # Test - _get_metrics
        rules = rules['superbench']['rules']
        for rule in ['rule0', 'rule1']:
            rulebase1._sb_rules[rule] = {}
            rulebase1._sb_rules[rule]['metrics'] = {}
            rulebase1._get_metrics(rule, rules)
            assert (len(rulebase1._sb_rules[rule]['metrics']) == 16)
