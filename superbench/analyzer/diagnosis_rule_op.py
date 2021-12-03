# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data diagnosis rule ops."""

from typing import Dict, Callable

import pandas as pd

from superbench.benchmarks.context import Enum
from superbench.common.utils import logger


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
    def check_criteria(cls, rule):
        """Get expression of the criteria and check if it's valid."""
        try:
            symbol = rule['criteria'].split(',')[0]
            condition = rule['criteria'].split(',')[1]
            if '%' in condition:
                condition = float(condition.strip('%')) / 100
            else:
                condition = float(condition)
            expression = symbol + str(condition)
            if not isinstance(eval('1' + expression), bool):
                logger.log_and_raise(exception=Exception, msg='invalid criteria format')
        except Exception as e:
            logger.log_and_raise(exception=Exception, msg='invalid criteria format - {}'.format(str(e)))
        return expression

    @classmethod
    def variance(cls, data_row, rule, summary_data_row, details, categories):
        """Rule op function of variance."""
        pass_rule = True
        # parse criteria and check if valid
        expression = RuleOp.check_criteria(rule)
        # every metric should pass the rule
        for metric in rule['metrics']:
            pass_metric = True
            # metric not in raw_data not the value is none, miss test
            if metric not in data_row or pd.isna(data_row[metric]):
                pass_rule = False
                details.append(metric + '_miss')
                categories.add(rule['categories'])
            else:
                # check if metric pass the rule
                val = data_row[metric]
                baseline = rule['metrics'][metric]
                if baseline == 0:
                    logger.log_and_raise(exception=Exception, msg='invalid baseline 0 in variance rule')
                var = (val - baseline) / baseline
                summary_data_row[metric] = var
                info = '(B/L: ' + '{:.4f}'.format(baseline) + ' VAL: ' + '{:.4f}'.format(val) + \
                    ' VAR: ' + '{:.4f}'.format(var * 100) + '%)'
                pass_metric = eval(str(var) + expression)
                # add issued details and categories
                if pass_metric is True:
                    pass_rule = False
                    details.append(metric + info)
                    categories.add(rule['categories'])
        return pass_rule

    @classmethod
    def value(cls, data_row, rule, summary_data_row, details, categories):
        """Rule op function of value higher than baseline."""
        pass_rule = True
        # parse criteria and check if valid
        expression = RuleOp.check_criteria(rule)
        # every metric should pass the rule
        for metric in rule['metrics']:
            pass_metric = True
            # metric not in raw_data not the value is none, miss test
            if metric not in data_row or pd.isna(data_row[metric]):
                pass_rule = False
                details.append(metric + '_miss')
                categories.add(rule['categories'])
            else:
                # check if metric pass the rule
                val = data_row[metric]
                summary_data_row[metric] = val
                condition = rule['criteria'].split(',')[1]
                info = '(B/L: ' + '{}'.format(condition) + ' VAL: ' + '{:.4f}'.format(val)
                pass_metric = eval(str(val) + expression)
                # add issued details and categories
                if pass_metric is True:
                    pass_rule = False
                    details.append(metric + info)
                    categories.add(rule['categories'])
        return pass_rule


RuleOp.add_rule_func(DiagnosisRuleType.VARIANCE)(RuleOp.variance)
RuleOp.add_rule_func(DiagnosisRuleType.VALUE)(RuleOp.value)
