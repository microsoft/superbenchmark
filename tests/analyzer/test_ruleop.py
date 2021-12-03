# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RuleOp module."""

import unittest

import pandas as pd

from superbench.analyzer import RuleOp, DiagnosisRuleType


class TestRuleOp(unittest.TestCase):
    """Test for Diagnosis Rule Ops."""
    def test_rule_op(self):
        """Test for defined rule operators."""
        # Test - get_rule_func
        # Negative case
        assert (not RuleOp.get_rule_func('fake'))
        # Positive case
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.VARIANCE)
        assert (rule_op == RuleOp.variance)

        # Test - check ciriteria
        false_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': '>',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': '5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': ',5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': '>,a',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }
        ]
        true_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': '>,50%',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2,
                    'kernel-launch/event_overhead:1': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': '<,-50%',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2,
                    'kernel-launch/event_overhead:1': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': '>,0',
                'function': 'value',
                'metrics': {
                    'kernel-launch/event_overhead:0': 0
                }
            }
        ]
        for rule in false_baselines:
            self.assertRaises(Exception, RuleOp.check_criteria, rule)
        for rule in true_baselines:
            assert (RuleOp.check_criteria(rule))

        # Test - rule function
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)

        # Test - variance
        data = {'kernel-launch/event_overhead:0': 3.1, 'kernel-launch/event_overhead:1': 2}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[0], summary_data_row, details, categories)
        assert (not pass_rule)

        data = {'kernel-launch/event_overhead:0': 1.5, 'kernel-launch/event_overhead:1': 1.5}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[1], summary_data_row, details, categories)
        assert (pass_rule)

        # Test - value
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.VALUE)
        pass_rule = rule_op(data_row, true_baselines[2], summary_data_row, details, categories)
        assert (not pass_rule)
