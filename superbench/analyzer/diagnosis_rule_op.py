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
    MULTI_RULES = 'multi_rules'


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

    @staticmethod
    def variance(data_row, rule, summary_data_row, details, categories):
        """Rule op function of variance.

        Each metric in the rule will calculate the variance (val - baseline / baseline),
        and use criteria in the rule to determine whether metric's variance meet the criteria,
        if any metric is labeled, the rule is not passed.

        Args:
            data_row (pd.Series): raw data of the metrics
            rule (dict): rule including function, criteria, metrics with their baseline values and categories
            summary_data_row (pd.Series): results of the metrics processed after the function
            details (list): details about violated rules and related data
            categories (set): categories of violated rules

        Returns:
            number: the number of the metrics that violate the rule if the rule is passed, otherwise 0
        """
        violated_metric_num = 0
        # parse criteria and check if valid
        if not isinstance(eval(rule['criteria'])(0), bool):
            logger.log_and_raise(exception=Exception, msg='invalid criteria format')
        # every metric should pass the rule
        for metric in rule['metrics']:
            violate_metric = False
            # metric not in raw_data or the value is none, miss test
            if metric not in data_row or pd.isna(data_row[metric]):
                violated_metric_num += 1
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
                violate_metric = eval(rule['criteria'])(var)
                # add issued details and categories
                if violate_metric:
                    violated_metric_num += 1
                    info = '(B/L: {:.4f} VAL: {:.4f} VAR: {:.2f}% Rule:{})'.format(
                        baseline, val, var * 100, rule['criteria']
                    )
                    details.append(metric + info)
                    categories.add(rule['categories'])
        return violated_metric_num

    @staticmethod
    def value(data_row, rule, summary_data_row, details, categories):
        """Rule op function of value.

        Each metric in the rule will use criteria in the rule
        to determine whether metric's value meet the criteria,
        if any metric is labeled, the rule is not passed.

        Args:
            data_row (pd.Series): raw data of the metrics
            rule (dict): rule including function, criteria, metrics with their baseline values and categories
            summary_data_row (pd.Series): results of the metrics processed after the function
            details (list): details about violated rules and related data
            categories (set): categories of violated rules

        Returns:
            number: the number of the metrics that violate the rule if the rule is passed, otherwise 0
        """
        violated_metric_num = 0
        # parse criteria and check if valid
        if not isinstance(eval(rule['criteria'])(0), bool):
            logger.log_and_raise(exception=Exception, msg='invalid criteria format')
        # every metric should pass the rule
        for metric in rule['metrics']:
            violate_metric = False
            # metric not in raw_data or the value is none, miss test
            if metric not in data_row or pd.isna(data_row[metric]):
                violated_metric_num += 1
                details.append(metric + '_miss')
                categories.add(rule['categories'])
            else:
                # check if metric pass the rule
                val = data_row[metric]
                summary_data_row[metric] = val
                violate_metric = eval(rule['criteria'])(val)
                # add issued details and categories
                if violate_metric:
                    violated_metric_num += 1
                    info = '(VAL: {:.4f} Rule:{})'.format(val, rule['criteria'])
                    details.append(metric + info)
                    categories.add(rule['categories'])
        return violated_metric_num

    @staticmethod
    def multi_rules(rule, details, categories, violation):
        """Rule op function of multi_rules.

        The criteria in this rule will use the combined results of multiple previous rules and their metrics
        which has been stored in advance to determine whether this rule is passed.

        Args:
            rule (dict): rule including function, criteria, metrics with their baseline values and categories
            details (list): details about violated rules and related data
            categories (set): categories of violated rules
            violation (dict): the number of the metrics that violate the rules
        Returns:
            number: 0 if the rule is passed, otherwise 1
        """
        violated = eval(rule['criteria'])(violation)
        if not isinstance(violated, bool):
            logger.log_and_raise(exception=Exception, msg='invalid upper criteria format')
        if violated:
            categories.add(rule['categories'])
            details.append('{}:{}'.format(rule['name'], rule['criteria']))
        return 1 if violated else 0


RuleOp.add_rule_func(DiagnosisRuleType.VARIANCE)(RuleOp.variance)
RuleOp.add_rule_func(DiagnosisRuleType.VALUE)(RuleOp.value)
RuleOp.add_rule_func(DiagnosisRuleType.MULTI_RULES)(RuleOp.multi_rules)
