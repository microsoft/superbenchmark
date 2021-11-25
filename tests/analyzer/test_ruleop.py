# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RuleOp module."""

import unittest

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
        # Test - variance
        # condition<0
        val = 2.5
        baseline = 2
        rule = {'name': 'variance', 'condition': -0.05}
        (pass_rule, var) = rule_op(val, baseline, rule)
        assert (not pass_rule)
        assert (var == -0.25)
        val = 1
        (pass_rule, var) = rule_op(val, baseline, rule)
        assert (pass_rule)
        assert (var == 0.5)
        # condition>0
        val = 1
        baseline = 2
        rule = {'name': 'variance', 'condition': 0.05}
        (pass_rule, var) = rule_op(val, baseline, rule)
        assert (not pass_rule)
        assert (var == 0.5)
        # invalid arguments
        baseline = 'a'
        self.assertRaises(Exception, rule_op, val, baseline, rule)
        baseline = 2
        rule = {'name': 'variance', 'condition': 'a'}
        self.assertRaises(Exception, rule_op, val, baseline, rule)
        # Test - higher_than_value
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.HIGHERTHANVALUE)
        assert (rule_op == RuleOp.higher_than_value)
        # val > baseline
        val = 2
        baseline = 1
        rule = {'name': 'value'}
        (pass_rule, var) = rule_op(val, baseline, rule)
        assert (not pass_rule)
        assert (var == 2)
        # val < baseline
        val = 1
        (pass_rule, var) = rule_op(val, baseline, rule)
        assert (pass_rule)
        assert (var == 1)
        # invalid argument
        baseline = 'a'
        self.assertRaises(Exception, rule_op, val, baseline, rule)
