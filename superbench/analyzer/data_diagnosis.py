# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for baseline-based data diagnosis."""
from typing import Callable
from pathlib import Path

import pandas as pd

from superbench.common.utils import logger
from superbench.analyzer.diagnosis_rule_op import RuleOp, DiagnosisRuleType
from superbench.analyzer import file_handler
from superbench.analyzer import RuleBase


class DataDiagnosis(RuleBase):
    """The DataDiagnosis class to do the baseline-based data diagnosis."""
    def __init__(self):
        """Init function."""
        super().__init__()

    def _check_and_format_rules(self, rule, name):
        """Check the rule of the metric whether the formart is valid.

        Args:
            rule (dict): the rule
            name (str): the rule name

        Returns:
            dict: the rule for the metric
        """
        # check if rule is supported
        super()._check_and_format_rules(rule, name)
        if 'function' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of function'.format(name))
        if not isinstance(DiagnosisRuleType(rule['function']), DiagnosisRuleType):
            logger.log_and_raise(exception=Exception, msg='{} invalid function name'.format(name))
        # check rule format
        if 'criteria' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of criteria'.format(name))
        if not isinstance(eval(rule['criteria']), Callable):
            logger.log_and_raise(exception=Exception, msg='invalid criteria format')
        if rule['function'] != 'multi_rules':
            if 'metrics' not in rule:
                logger.log_and_raise(exception=Exception, msg='{} lack of metrics'.format(name))
        if 'store' in rule and not isinstance(rule['store'], bool):
            logger.log_and_raise(exception=Exception, msg='{} store must be bool type'.format(name))
        return rule

    def _get_baseline_of_metric(self, baseline, metric):
        """Get the baseline value of the metric.

        Args:
            baseline (dict): baseline defined in baseline file
            metric (str): the full name of the metric

        Returns:
            numeric: the baseline value of the metric
        """
        if metric in baseline:
            return baseline[metric]
        else:
            # exclude rank info
            short = metric.split(':')[0]
            if short in baseline:
                return baseline[short]
            # baseline not defined
            else:
                logger.warning('DataDiagnosis: get baseline - {} baseline not found'.format(metric))
                return -1

    def __get_metrics_and_baseline(self, rule, benchmark_rules, baseline):
        """Get metrics with baseline in the rule.

        Parse metric regex in the rule, and store the (baseline, metric) pair
        in _sb_rules[rule]['metrics'] and metric in _enable_metrics。

        Args:
            rule (str): the name of the rule
            benchmark_rules (dict): the dict of rules
            baseline (dict): the dict of baseline of metrics
        """
        if 'function' in self._sb_rules[rule] and self._sb_rules[rule]['function'] == 'multi_rules':
            return
        self._get_metrics(rule, benchmark_rules)
        for metric in self._sb_rules[rule]['metrics']:
            self._sb_rules[rule]['metrics'][metric] = self._get_baseline_of_metric(baseline, metric)

    def _parse_rules_and_baseline(self, rules, baseline):
        """Parse and merge rules and baseline read from file.

        Args:
            rules (dict): rules from rule yaml file
            baseline (dict): baseline of metrics from baseline json file

        Returns:
            bool: return True if successfully get the criteria for all rules, otherwise False.
        """
        try:
            if not rules:
                logger.error('DataDiagnosis: get criteria failed')
                return False
            self._sb_rules = {}
            self._enable_metrics = set()
            benchmark_rules = rules['superbench']['rules']
            for rule in benchmark_rules:
                benchmark_rules[rule] = self._check_and_format_rules(benchmark_rules[rule], rule)
                self._sb_rules[rule] = {}
                self._sb_rules[rule]['name'] = rule
                self._sb_rules[rule]['function'] = benchmark_rules[rule]['function']
                self._sb_rules[rule]['store'] = True if 'store' in benchmark_rules[
                    rule] and benchmark_rules[rule]['store'] is True else False
                self._sb_rules[rule]['criteria'] = benchmark_rules[rule]['criteria']
                self._sb_rules[rule]['categories'] = benchmark_rules[rule]['categories']
                self._sb_rules[rule]['metrics'] = {}
                self.__get_metrics_and_baseline(rule, benchmark_rules, baseline)
            self._enable_metrics = sorted(list(self._enable_metrics))
        except Exception as e:
            logger.error('DataDiagnosis: get criteria failed - {}'.format(str(e)))
            return False

        return True

    def _run_diagnosis_rules_for_single_node(self, node):
        """Use rules to diagnosis single node data.

        Use the rules defined in rule_file to diagnose the raw data of each node,
        if the node violate any rule, label as defective node and save
        the 'Category', 'Defective Details' and data summary of defective node.

        Args:
            node (str): the node to do the diagosis

        Returns:
            details_row (list): None if the node is not labeled as defective,
                otherwise details of ['Category', 'Defective Details']
            summary_data_row (dict): None if the node is not labeled as defective,
                otherwise data summary of the metrics
        """
        data_row = self._raw_data_df.loc[node]
        issue_label = False
        details = []
        categories = set()
        violation = {}
        summary_data_row = pd.Series(index=self._enable_metrics, name=node, dtype=float)
        # Check each rule
        for rule in self._sb_rules:
            # Get rule op function and run the rule
            function_name = self._sb_rules[rule]['function']
            rule_op = RuleOp.get_rule_func(DiagnosisRuleType(function_name))
            violated_num = 0
            if rule_op == RuleOp.multi_rules:
                violated_num = rule_op(self._sb_rules[rule], details, categories, violation)
            else:
                violated_num = rule_op(data_row, self._sb_rules[rule], summary_data_row, details, categories)
            # label the node as defective one
            if self._sb_rules[rule]['store']:
                violation[rule] = violated_num
            elif violated_num:
                issue_label = True
        if issue_label:
            # Add category information
            general_cat_str = ','.join(sorted(list(categories)))
            details_cat_str = ','.join(sorted((details)))
            details_row = [general_cat_str, details_cat_str]
            return details_row, summary_data_row

        return None, None

    def run_diagnosis_rules(self, rules, baseline):
        """Rule-based data diagnosis for multiple nodes' raw data.

        Use the rules defined in rules to diagnose the raw data of each node,
        if the node violate any rule, label as defective node and save
        the 'Category', 'Defective Details' and processed data of defective node.

        Args:
            rules (dict): rules from rule yaml file
            baseline (dict): baseline of metrics from baseline json file

        Returns:
            data_not_accept_df (DataFrame): defective nodes's detailed information
            label_df (DataFrame): labels for all nodes
        """
        try:
            summary_columns = ['Category', 'Defective Details']
            data_not_accept_df = pd.DataFrame(columns=summary_columns)
            summary_details_df = pd.DataFrame()
            label_df = pd.DataFrame(columns=['label'])
            if not self._parse_rules_and_baseline(rules, baseline):
                return data_not_accept_df, label_df
            # run diagnosis rules for each node
            for node in self._raw_data_df.index:
                details_row, summary_data_row = self._run_diagnosis_rules_for_single_node(node)
                if details_row:
                    data_not_accept_df.loc[node] = details_row
                    summary_details_df = summary_details_df.append(summary_data_row)
                    label_df.loc[node] = 1
                else:
                    label_df.loc[node] = 0
            # combine details for defective nodes
            if len(data_not_accept_df) != 0:
                data_not_accept_df = data_not_accept_df.join(summary_details_df)
                data_not_accept_df = data_not_accept_df.sort_values(by=summary_columns, ascending=False)

        except Exception as e:
            logger.error('DataDiagnosis: run diagnosis rules failed, message: {}'.format(str(e)))
        return data_not_accept_df, label_df

    def run(self, raw_data_file, rule_file, baseline_file, output_dir, output_format='excel'):
        """Run the data diagnosis and output the results.

        Args:
            raw_data_file (str): the path of raw data jsonl file.
            rule_file (str): The path of baseline yaml file
            baseline_file (str): The path of baseline json file
            output_dir (str): the directory of output file
            output_format (str): the format of the output, 'excel' or 'json'
        """
        try:
            rules = self._preprocess(raw_data_file, rule_file)
            # read baseline
            baseline = file_handler.read_baseline(baseline_file)
            logger.info('DataDiagnosis: Begin to process {} nodes'.format(len(self._raw_data_df)))
            data_not_accept_df, label_df = self.run_diagnosis_rules(rules, baseline)
            logger.info('DataDiagnosis: Processed finished')
            output_path = ''
            if output_format == 'excel':
                output_path = str(Path(output_dir) / 'diagnosis_summary.xlsx')
                file_handler.output_excel(self._raw_data_df, data_not_accept_df, output_path, self._sb_rules)
            elif output_format == 'json':
                output_path = str(Path(output_dir) / 'diagnosis_summary.jsonl')
                file_handler.output_json_data_not_accept(data_not_accept_df, output_path)
            else:
                logger.error('DataDiagnosis: output failed - unsupported output format')
            logger.info('DataDiagnosis: Output results to {}'.format(output_path))
        except Exception as e:
            logger.error('DataDiagnosis: run failed - {}'.format(str(e)))
