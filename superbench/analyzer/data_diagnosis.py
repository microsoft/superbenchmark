# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import re
from typing import Callable

import pandas as pd

from superbench.common.utils import logger
from superbench.analyzer.diagnosis_rule_op import RuleOp, DiagnosisRuleType
import superbench.analyzer.file_handler as file_handler


class DataDiagnosis():
    """The DataDiagnosis class to do the baseline-based data diagnosis."""
    def __init__(self):
        """Init function using the raw data file path."""
        self._sb_rules = {}
        self._metrics = {}

    def _get_metrics_by_benchmarks(self, metrics_list):
        """Get all metrics by benchmark from metrics_list."""
        benchmarks_metrics = {}
        for metric in metrics_list:
            benchmark = metric.split('/')[0]
            if benchmark not in benchmarks_metrics:
                benchmarks_metrics[benchmark] = set()
            benchmarks_metrics[benchmark].add(metric)
        return benchmarks_metrics

    def _check_rules(self, rule, name):
        """Check the rule of the metric whether formart is valid.

        Args:
            rule (dict): the rule
            name ([type]): the rule name

        Returns:
            dict: criteria and rule for the metric
        """
        # check if rule is supported
        if not isinstance(DiagnosisRuleType(rule['function']), DiagnosisRuleType):
            logger.log_and_raise(exception=Exception, msg='{} invalid function name'.format(name))
        # check rule format
        if 'criteria' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of criteria'.format(name))
        if not isinstance(eval(rule['criteria']), Callable):
            logger.log_and_raise(exception=Exception, msg='invalid criteria format')
        if 'categories' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of category'.format(name))
        if 'metrics' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of metrics'.format(name))
        if isinstance(rule['metrics'], str):
            rule['metrics'] = [rule['metrics']]
        return rule

    def _get_criteria(self, rule_file, baseline_file):
        """Get and generate criteria from metrics.

        Use metric with regex in rule_file to match the metric full name from raw data
        for each benchmark enabled in baseline file, and then generate criteria and rule for
        matched metrics.

        Args:
            rule_file (str): The path of rule yaml file
            baseline_file (str): The path of baseline json file

        Returns:
            bool: return True if successfully get the criteria, otherwise False.
        """
        try:
            rules = file_handler.read_rules(rule_file)
            baseline = file_handler.read_baseline(baseline_file)
            if not rules or not baseline:
                return False
            self._sb_rules = {}
            self._enable_metrics = []
            benchmark_rules = rules['superbench']['rules']
            for rule in benchmark_rules:
                benchmark_rules[rule] = self._check_rules(benchmark_rules[rule], rule)
                self._sb_rules[rule] = {}
                self._sb_rules[rule]['function'] = benchmark_rules[rule]['function']
                self._sb_rules[rule]['criteria'] = benchmark_rules[rule]['criteria']
                self._sb_rules[rule]['categories'] = benchmark_rules[rule]['categories']
                self._sb_rules[rule]['metrics'] = {}
                single_rule_metrics = benchmark_rules[rule]['metrics']
                benchmark_metrics = self._get_metrics_by_benchmarks(single_rule_metrics)
                for benchmark_name in benchmark_metrics:
                    # get rules and criteria for each metric
                    for metric in self._metrics[benchmark_name]:
                        # metric full name in baseline
                        if metric in single_rule_metrics:
                            self._sb_rules[rule]['metrics'][metric] = baseline[metric]
                            self._enable_metrics.append(metric)
                            continue
                        # metric full name not in baseline, use regex to match
                        for metric_regex in benchmark_metrics[benchmark_name]:
                            if re.search(metric_regex, metric):
                                self._sb_rules[rule]['metrics'][metric] = baseline[metric]
                                self._enable_metrics.append(metric)
        except Exception as e:
            logger.error('DataDiagnosis: get criteria failed- {}'.format(str(e)))
            return False

        return True

    def _run_diagnosis_rules_for_single_node(self, node):
        """Use rules to diagnosis single node data.

        Use the rules defined in rule_file to diagnose the raw data of each node,
        if the node violate any rule, label as issued node and save
        the '# of Issues', 'Category', 'Issue Details' and processed data of issued node.

        Args:
            node (str): the node to do the diagosis

        Returns:
            details_row (list): None if the node is not labeled as issued,
                otherwise details of ['# of Issues', 'Category', 'Issue Details']
            summary_data_row (dict): None if the node is not labeled as issued,
                otherwise data summary of the metrics used in diagnosis
        """
        data_row = self._raw_data_df.loc[node]
        issue_label = False
        details = []
        categories = set()
        summary_data_row = pd.Series(index=self._enable_metrics, name=node, dtype=float)

        for rule in self._sb_rules:
            # Get rule op function and run the rule
            function_name = self._sb_rules[rule]['function']
            rule_op = RuleOp.get_rule_func(DiagnosisRuleType(function_name))
            pass_rule = rule_op(data_row, self._sb_rules[rule], summary_data_row, details, categories)
            # label the node as issued one
            if not pass_rule:
                issue_label = True
        if issue_label:
            # Add category information
            general_cat_str = ','.join(categories)
            details_cat_str = ','.join(details)
            details_row = [len(details), general_cat_str, details_cat_str]
            return details_row, summary_data_row

        return None, None

    def run_diagnosis_rules(self, rule_file, baseline_file):
        """Rule-based data diagnosis for multi nodes' raw data.

        Use the rules defined in rule_file to diagnose the raw data of each node,
        if the node violate any rule, label as issued node and save
        the '# of Issues', 'Category', 'Issue Details' and processed data of issued node.

        Args:
            rule_file (str): The path of rule yaml file
            baseline_file (str): The path of baseline json file

        Returns:
            data_not_accept_df (DataFrame): issued nodes's detailed information
            label_df (DataFrame): labels for all nodes
        """
        try:
            summary_columns = ['# of Issues', 'Category', 'Issue Details']
            data_not_accept_df = pd.DataFrame(columns=summary_columns)
            summary_details_df = pd.DataFrame()
            label_df = pd.DataFrame(columns=['label'])
            # check raw data whether empty
            if len(self._raw_data_df) == 0:
                logger.error('DataDiagnosis: empty raw data')
                return data_not_accept_df, label_df
            # get criteria
            if not self._get_criteria(rule_file, baseline_file):
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
            # combine details for issued nodes
            if len(data_not_accept_df) != 0:
                data_not_accept_df = data_not_accept_df.join(summary_details_df)
                data_not_accept_df = data_not_accept_df.sort_values(by=summary_columns, ascending=False)

        except Exception as e:
            logger.error('DataDiagnosis: run diagnosis rules failed, message: {}'.format(str(e)))
        return data_not_accept_df, label_df

    def excel_output(self, data_not_accept_df, output_dir):
        """Output the processed results into excel file.

        Args:
            data_not_accept_df (DataFrame): issued nodes's detailed information
            output_dir (str): the path of output excel file
        """
        try:
            writer = pd.ExcelWriter(output_dir + '/diagnosis_summary.xlsx', engine='xlsxwriter')
            # Check whether writer is valiad
            if not isinstance(writer, pd.ExcelWriter):
                logger.error('DataDiagnosis: excel_data_output - invalid file path.')
                return
            file_handler.excel_raw_data_output(writer, self._raw_data_df, 'Raw Data')
            file_handler.excel_data_not_accept_output(writer, data_not_accept_df, self._sb_rules)
            writer.save()
        except Exception as e:
            logger.error('DataDiagnosis: excel_data_output - {}'.format(str(e)))

    def run(self, raw_data_path, rule_file, output_dir, output_format='excel'):
        """Run the data diagnosis and output the results.

        Args:
            raw_data_path (str): the path of raw data jsonl file.
            rule_file (str): The path of baseline yaml file
            output_dir (str): the path of output excel file
            output_format (str): the format of the output, 'excel' or 'json'
        """
        try:
            self._raw_data_df = file_handler.read_raw_data(raw_data_path)
            self._metrics = self._get_metrics_by_benchmarks(list(self._raw_data_df.columms))
            logger.info('DataDiagnosis: Begin to processe {} nodes'.format(len(self._raw_data_df)))
            data_not_accept_df, label_df = self.run_diagnosis_rules(self, rule_file)
            logger.info('DataDiagnosis: Processed finished')
            if output_format == 'excel':
                self.excel_output(data_not_accept_df, output_dir)
            elif output_format == 'json':
                data_not_accept_df.to_json(output_dir + '/diagnosis_summary.json')
            else:
                logger.error('DataDiagnosis: output failed - unsupported output format')
        except Exception as e:
            logger.error('DataDiagnosis: run failed - {}'.format(str(e)))
