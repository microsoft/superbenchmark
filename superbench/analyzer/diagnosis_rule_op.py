# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data diagnosis rule ops."""

from typing import Dict, Callable

from superbench.benchmarks.context import Enum


class DiagnosisRuleType(Enum):
    """The Enum class representing different rule ops."""

    VARIANCE = 'variance'
    HIGHERTHANVALUE = 'value'


class RuleOp:
    """RuleOp class to maintain all rule functions."""

    functions: Dict[DiagnosisRuleType, Callable] = dict()

    @classmethod
    def add_rule_func(cls, rule_type):
        """Add rule fuction.

        Args:
            rule_type (DiagnosisRuleType): The type of rule function.

        Return:
            decorator (Callable): return the decorator to add the rule function.
        """
        def decorator(func):
            cls.functions[rule_type] = func
            return func

        return decorator

    @classmethod
    def get_rule_func(cls, rule_type):
        """Get rule fuction by rule_type.

        Args:
            rule_type (DiagnosisRuleType): The type of rule function.

        Return:
            func (Callable): rule function, None means invalid rule type.
        """
        if rule_type in cls.functions:
            return cls.functions[rule_type]

        return None

    @classmethod
    def variance(cls, val, baseline, rule):
        """Rule op function of variance."""
        pass_rule = True
        var = 1 - val / baseline
        if 'condition' not in rule:
            rule['condition'] = -0.05
        if rule['condition'] >= 0:
            if var > rule['condition']:
                pass_rule = False
        else:
            if var < rule['condition']:
                pass_rule = False
        return (pass_rule, var)

    @classmethod
    def higher_than_value(cls, val, baseline, rule):
        """Rule op function of value higher than baseline."""
        pass_rule = True
        if val > baseline:
            pass_rule = False
        return (pass_rule, val)


RuleOp.add_rule_func(DiagnosisRuleType.VARIANCE)(RuleOp.variance)
RuleOp.add_rule_func(DiagnosisRuleType.HIGHERTHANVALUE)(RuleOp.higher_than_value)
