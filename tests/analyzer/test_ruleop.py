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
        true_baselines = [
            {
                'categories': 'KernelLaunch',
                'criteria': '>,50%',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
                }
            }, {
                'categories': 'KernelLaunch',
                'criteria': '<,-50%',
                'function': 'variance',
                'metrics': {
                    'kernel-launch/event_overhead:0': 2
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
        details = []
        categories = set()
        summary_data_row = pd.Series(index=['kernel-launch/event_overhead:0'], dtype=float)

        # Test - variance
        data = {'kernel-launch/event_overhead:0': 3.1}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[0], summary_data_row, details, categories)
        assert (not pass_rule)

        data = {'kernel-launch/event_overhead:0': 1.5}
        data_row = pd.Series(data)
        pass_rule = rule_op(data_row, true_baselines[1], summary_data_row, details, categories)
        assert (pass_rule)

        # Test - value
        pass_rule = rule_op(data_row, true_baselines[2], summary_data_row, details, categories)
        assert (not pass_rule)
