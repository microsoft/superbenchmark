# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import numbers
import re

import pandas as pd

from superbench.common.utils import logger
from superbench.analyzer.diagnosis_rule_op import RuleOp, DiagnosisRuleType
import superbench.analyzer.file_handler as file_handler


class DataDiagnosis():
    """The DataDiagnosis class to process and analyze the sammary data."""
    def __init__(self, raw_data_path):
        """Init function using the raw data file path.

        Args:
            raw_data_path (str): the path of raw data jsonl file.
        """
        self._raw_data_path = raw_data_path
        self._sb_baseline = {}
        self._metrics = {}
        self._raw_data_df = file_handler.read_raw_data(self._raw_data_path)
        self._get_metrics_from_raw_data()

    def _get_metrics_from_raw_data(self):
        """Get all metrics by benchmark from raw data."""
        if len(self._raw_data_df) == 0:
            return
        for metric in self._raw_data_df.columns:
            benchmark = metric.split('/')[0]
            if benchmark not in self._metrics:
                self._metrics[benchmark] = set()
            self._metrics[benchmark].add(metric)

    def __get_validation_enabled_benchmarks(self):
        """Get validation enabled benchmarks list.

        Return:
            list: List of benchmarks which will be executed.
        """
        if self._sb_baseline['superbench']['enable']:
            if isinstance(self._sb_baseline['superbench']['enable'], str):
                return [self._sb_baseline['superbench']['enable']]
            elif isinstance(self._sb_baseline['superbench']['enable'], list):
                return list(self._sb_baseline['superbench']['enable'])
        return [k for k, v in self._sb_baseline['superbench']['benchmarks'].items() if v['enable']]

    def _check_baseline(self, baseline, metric):
        """Check the baseline of the metric whether formart is valid.

        Args:
            baseline (dict): criteria and rule for the metric
            metric ([type]): the metric name

        Returns:
            dict: criteria and rule for the metric
        """
        # use mean of the column as default criteria if not set
        if 'criteria' not in baseline:
            baseline['criteria'] = self._raw_data_df[metric].mean()
            logger.warning('DataDiagnosis: get_criteria - [{}] use the mean when criteria is not set.'.format(metric))
        # check if criteria is number
        if not isinstance(baseline['criteria'], numbers.Number):
            logger.log_and_raise(exception=Exception, msg='{} invalid criteria'.format(metric))
        # check if rule is supported
        if not isinstance(DiagnosisRuleType(baseline['rules']['name']), DiagnosisRuleType):
            logger.log_and_raise(exception=Exception, msg='{} invalid rule name'.format(metric))
        # check if variance rule is valid
        if baseline['rules']['name'] == 'variance':
            if not isinstance(baseline['rules']['condition'], numbers.Number):
                logger.log_and_raise(exception=Exception, msg='{} invalid variance rule'.format(metric))
        return baseline

    def _get_criteria(self, baseline_file):
        """Get and generate criteria from metrics.

        Use metric with regex in baseline_file to match the metric full name from raw data
        for each benchmark enabled in baseline file, and then generate criteria and rule for
        matched metrics.

        Args:
            baseline_file (str): The path of baseline yaml file

        Returns:
            bool: return True if successfully get the criteria, otherwise False.
        """
        try:
            self._sb_baseline = file_handler.read_baseline(baseline_file)
            if not self._sb_baseline:
                return False
            full_baseline = {}
            self._sb_enable_validation_benchmarks = self.__get_validation_enabled_benchmarks()
            benchmark_rules = self._sb_baseline['superbench']['benchmarks']
            for benchmark_name in self._sb_enable_validation_benchmarks:
                single_benchmark_rules = benchmark_rules[benchmark_name]['metrics']
                # return_code rule should be a default rule for each benchmark in rule file
                # if all nodes miss the benchmark and no return code, manually add it
                if benchmark_name not in self._metrics:
                    self._metrics[benchmark_name] = {benchmark_name + '/return_code'}
                # get rules and criteria for each metric
                for metric in self._metrics[benchmark_name]:
                    # metric full name in baseline
                    if metric in single_benchmark_rules:
                        full_baseline[metric] = single_benchmark_rules[metric]
                        continue
                    # metric full name not in baseline, use regex to match
                    for rule_metric in single_benchmark_rules:
                        if re.search(rule_metric, metric):
                            full_baseline[metric] = single_benchmark_rules[rule_metric]
                            full_baseline[metric] = self._check_baseline(full_baseline[metric], metric)
                            break
            self._sb_baseline = full_baseline
        except Exception as e:
            logger.error('DataDiagnosis: invalid rule file fomat - {}'.format(str(e)))
            return False

        return True

    def hw_issue(self, benchmark_list):
        """Idendify if the benchmark is classified as hardware issue.

        All benchmarks except models and inferences(TensorRT and ORT) are classified hardware issue.

        Args:
            benchmark_list (list): list of benchmarks

        Returns:
            bool: return true if it's hardware issue
        """
        if not isinstance(benchmark_list, set):
            return False
        for category in benchmark_list:
            if 'models' not in category and 'inference' not in category:
                return True
        return False

    def _run_diagnosis_rules_for_single_node(self, node):
        """Use rules to diagnosis single node data.

        Use the rules defined in rule_file to diagnose the raw data of each node,
        if the node violate any rule, label as issued node and save
        the 'Hw Issues', '# of Issues', 'Category', 'Issue Details' and processed data of issued node.

        Args:
            node (str): the node to do the diagosis

        Returns:
            details_row (list): None if the node is not labeled as issued,
                otherwise details of ['Hw Issues', '# of Issues', 'Category', 'Issue Details']
            summary_data_row (dict): None if the node is not labeled as issued,
                otherwise data summary of the metrics used in diagnosis
        """
        data_row = self._raw_data_df.loc[node]
        columns = self._raw_data_df.columns
        issue_label = False
        category_details = []
        categories = set()
        summary_data_row = pd.Series(index=self._sb_baseline.keys(), name=node, dtype=float)
        cnn_benchmarks = ['densenet_models', 'vgg_models', 'resnet_models']
        model_label_num = 0

        for metric in self._sb_baseline:
            benchmark = metric.split('/')[0]
            baseline = self._sb_baseline[metric]['criteria']
            pass_rule = True
            # metric not in raw_data not the value is none, miss test
            if metric not in columns or pd.isna(data_row[metric]):
                pass_rule = False
            # check if metric pass the rule
            else:
                data = data_row[metric]
                rule = self._sb_baseline[metric]['rules']
                rule_op = RuleOp.get_rule_func(DiagnosisRuleType(rule['name']))
                (pass_rule, processed) = rule_op(data, baseline, rule)
                summary_data_row[metric] = processed
            # label the node as issued one
            if not pass_rule:
                # use return code to identify 'miss test'
                if 'return_code' in metric:
                    category_details.append(benchmark + '_miss')
                    categories.add('MissTest')
                    issue_label = True
                # use isna to identify miss metrics
                elif pd.isna(data_row[metric]):
                    category_details.append(metric + '_miss')
                    categories.add('MissTest')
                    issue_label = True
                else:
                    category_details.append(metric)
                    categories.add(benchmark)
                    if benchmark not in cnn_benchmarks:
                        issue_label = True
                    # for cnn models, only 2 model metric issued, label as issue
                    else:
                        model_label_num += 1
                        if model_label_num >= 2:
                            issue_label = True

        if issue_label:
            # Add category information
            general_cat_str = ','.join(categories)
            details_cat_str = ','.join(category_details)
            hw_issue_flag = self.hw_issue(categories)
            details_row = [hw_issue_flag, len(category_details), general_cat_str, details_cat_str]
            return details_row, summary_data_row

        return None, None

    def run_diagnosis_rules(self, rule_file):
        """Rule-based data diagnosis for multi nodes' raw data.

        Use the rules defined in rule_file to diagnose the raw data of each node,
        if the node violate any rule, label as issued node and save
        the 'Hw Issues', '# of Issues', 'Category', 'Issue Details' and processed data of issued node.

        Args:
            rule_file (str): The path of baseline yaml file

        Returns:
            data_not_accept_df (DataFrame): issued nodes's detailed information
            label_df (DataFrame): labels for all nodes
        """
        try:
            summary_columns = ['Hw Issues', '# of Issues', 'Category', 'Issue Details']
            data_not_accept_df = pd.DataFrame(columns=summary_columns)
            summary_details_df = pd.DataFrame()
            label_df = pd.DataFrame(columns=['label'])
            # check raw data whether empty
            if len(self._raw_data_df) == 0:
                logger.error('DataDiagnosis: empty raw data')
                return data_not_accept_df, label_df
            # get criteria
            if not self._get_criteria(rule_file):
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

    def excel_output(self, data_not_accept_df, output_file):
        """Output the processed results into excel file.

        Args:
            data_not_accept_df (DataFrame): issued nodes's detailed information
            output_file (str): the path of output excel file
        """
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        # Check whether writer is valiad
        if not isinstance(writer, pd.ExcelWriter):
            logger.error('DataDiagnosis: excel_data_output - invalid file path.')
            return
        file_handler.excel_raw_data_output(writer, self._raw_data_df, 'Raw Data')
        file_handler.excel_data_not_accept_output(writer, data_not_accept_df, self._sb_baseline)
        writer.save()
