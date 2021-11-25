# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RuleOp module."""

import unittest

from superbench.analyzer import RuleOp, DiagnosisRuleType


class TestRuleOp(unittest.TestCase):
    """Test for Diagnosis Rule Ops."""
    def test_rule_op(self):
        """Test for defined rule operators."""
        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.VARIANCE)
        assert (rule_op == RuleOp.variance)
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
        baseline = 'a'
        self.assertRaises(Exception, rule_op, val, baseline, rule)
        baseline = 2
        rule = {'name': 'variance', 'condition': 'a'}
        self.assertRaises(Exception, rule_op, val, baseline, rule)

        rule_op = RuleOp.get_rule_func(DiagnosisRuleType.HIGHERTHANVALUE)
        assert (rule_op == RuleOp.higher_than_value)
        val = 2
        baseline = 1
        rule = {'name': 'value'}
        (pass_rule, var) = rule_op(val, baseline, rule)
        assert (not pass_rule)
        assert (var == 2)
        val = 1
        (pass_rule, var) = rule_op(val, baseline, rule)
        assert (pass_rule)
        assert (var == 1)
        baseline = 'a'
        self.assertRaises(Exception, rule_op, val, baseline, rule)
