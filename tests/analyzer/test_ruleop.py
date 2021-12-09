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

        # Test - variance and value rule function
        # Check whether arguments are valid
        # Negative case
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)
        data = {'kernel-launch/event_overhead:0': 3.1, 'kernel-launch/event_overhead:1': 2}
        data_row = pd.Series(data)
        false_rule_and_baselines = [
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

        for rule in false_rule_and_baselines:
            self.assertRaises(Exception, RuleOp.variance, data_row, rule, summary_data_row, details, categories)
            self.assertRaises(Exception, RuleOp.value, data_row, rule, summary_data_row, details, categories)

        # Positive case
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
                'categories': 'KernelLaunch2',
                'criteria': 'lambda x:x>0',
                'function': 'value',
                'metrics': {
                    'kernel-launch/event_overhead:0': 0
                }
            }
        ]
        # Check results
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)
        # variance
        data = {'kernel-launch/event_overhead:0': 3.1, 'kernel-launch/event_overhead:1': 2}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[0], summary_data_row, details, categories)
        assert (not pass_rule)
        assert (categories == {'KernelLaunch'})
        assert (details == ['kernel-launch/event_overhead:0(B/L: 2.0000 VAL: 3.1000 VAR: 55.00% Rule:lambda x:x>0.5)'])

        data = {'kernel-launch/event_overhead:0': 1.5, 'kernel-launch/event_overhead:1': 1.5}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[1], summary_data_row, details, categories)
        assert (pass_rule)
        assert (categories == {'KernelLaunch'})

        # value
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.VALUE)
        pass_rule = rule_op(data_row, true_baselines[2], summary_data_row, details, categories)
        assert (not pass_rule)
        assert (categories == {'KernelLaunch', 'KernelLaunch2'})
        assert ('kernel-launch/event_overhead:0(VAL: 1.5000 Rule:lambda x:x>0)' in details)
        assert ('kernel-launch/event_overhead:0(B/L: 2.0000 VAL: 3.1000 VAR: 55.00% Rule:lambda x:x>0.5)' in details)
