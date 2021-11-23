# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for RuleOp module."""

from superbench.analyzer import RuleOp, RuleType


def test_rule_op():
    """Test for defined rule operators."""
    rule_op = RuleOp.get_rule_func(RuleType.VARIANCE)
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

    rule_op = RuleOp.get_rule_func(RuleType.HIGHERTHANVALUE)
    assert (rule_op == RuleOp.higher_than_value)
    val = 2
    baseline = 1
    rule = {'name': 'highervalue'}
    (pass_rule, var) = rule_op(val, baseline, rule)
    assert (not pass_rule)
    assert (var == 2)
    val = 1
    (pass_rule, var) = rule_op(val, baseline, rule)
    assert (pass_rule)
    assert (var == 1)
