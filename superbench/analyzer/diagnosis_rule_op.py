# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data diagnosis rule ops."""

from typing import Dict, Callable

import pandas as pd

from superbench.benchmarks.context import Enum


class DiagnosisRuleType(Enum):
    """The Enum class representing different rule ops."""

    VARIANCE = 'variance'
    VALUE = 'value'


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
    def variance(cls, data_row, rule, summary_data_row, details, categories):
        """Rule op function of variance."""
        pass_rule = True
        # parse symbol and condition
        symbol = rule['criteria'].split(',')[0]
        condition = rule['criteria'].split(',')[1]
        if '%' in condition:
            condition = float(condition.strip('%')) / 100
        else:
            condition = float(condition)
        # every metric should pass the rule
        for metric in rule['metrics']:
            pass_metric = True
            # metric not in raw_data not the value is none, miss test
            if metric not in data_row or pd.isna(data_row[metric]):
                pass_rule = False
                details.append(metric + '_miss')
                categories.add(rule['categories'])
                continue
            # check if metric pass the rule
            val = data_row[metric]
            baseline = rule['metrics'][metric]
            var = (val - baseline) / baseline
            summary_data_row[metric] = var
            info = '(B/L: ' + '{:.4f}'.format(baseline) + ' VAL: ' + '{:.4f}'.format(val) + \
                ' VAR: ' + '{:.4f}'.format(var * 100) + '%)'
            if symbol == '>':
                if var > condition:
                    pass_metric = False
            elif symbol == '<':
                if var < condition:
                    pass_metric = False
            # add issued details and categories
            if not pass_metric:
                pass_rule = False
                details.append(metric + info)
                categories.add(rule['categories'])
        return pass_rule

    @classmethod
    def value(cls, data_row, rule, summary_data_row, details, categories):
        """Rule op function of value higher than baseline."""
        pass_rule = True
        # parse symbol and condition
        symbol = rule['criteria'].split(',')[0]
        condition = float(rule['criteria'].split(',')[1])
        # every metric should pass the rule
        for metric in rule['metrics']:
            pass_metric = True
            # metric not in raw_data not the value is none, miss test
            if metric not in data_row or pd.isna(data_row[metric]):
                pass_rule = False
                details.append(metric + '_miss')
                categories.add(rule['categories'])
                continue
            # check if metric pass the rule
            val = data_row[metric]
            summary_data_row[metric] = val
            info = '(B/L: ' + '{:.4f}'.format(condition) + ' VAL: ' + '{:.4f}'.format(val)
            if symbol == '>':
                if val > condition:
                    pass_metric = False
            elif symbol == '<':
                if val < condition:
                    pass_metric = False
            # add issued details and categories
            if not pass_metric:
                pass_rule = False
                details.append(metric + info)
                categories.add(rule['categories'])
        return pass_rule


RuleOp.add_rule_func(DiagnosisRuleType.VARIANCE)(RuleOp.variance)
RuleOp.add_rule_func(DiagnosisRuleType.VALUE)(RuleOp.value)
