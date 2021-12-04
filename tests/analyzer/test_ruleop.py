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
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)
        data = {'kernel-launch/event_overhead:0': 3.1, 'kernel-launch/event_overhead:1': 2}
        data_row = pd.Series(data)

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
                'criteria': '>5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x+1',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }
        ]

        for rule in false_baselines:
            self.assertRaises(Exception, RuleOp.variance, data_row, rule, summary_data_row, details, categories)
            self.assertRaises(Exception, RuleOp.value, data_row, rule, summary_data_row, details, categories)

        # Test - rule function
        true_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0.5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2,
                    'kernel-launch/event_overhead:1': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x<-0.5',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2,
                    'kernel-launch/event_overhead:1': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': 'lambda x:x>0',
                'function': 'value',
                'metrics': {
                    'kernel-launch/event_overhead:0': 0
                }
            }
        ]
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)

        # variance
        data = {'kernel-launch/event_overhead:0': 3.1, 'kernel-launch/event_overhead:1': 2}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[0], summary_data_row, details, categories)
        assert (not pass_rule)

        data = {'kernel-launch/event_overhead:0': 1.5, 'kernel-launch/event_overhead:1': 1.5}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[1], summary_data_row, details, categories)
        assert (pass_rule)

        # value
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.VALUE)
        pass_rule = rule_op(data_row, true_baselines[2], summary_data_row, details, categories)
        assert (not pass_rule)
