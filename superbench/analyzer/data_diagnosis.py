# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import re
from pathlib import Path

import jsonlines
import pandas as pd
import yaml

from superbench.common.utils import logger
from superbench.analyzer.rule_op import RuleOp, RuleType


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
        self._raw_data_df = pd.DataFrame()
        self._read_raw_data(self._raw_data_path)
        self._get_metrics_from_raw_data()

    def _read_raw_data(self, raw_data_path):
        """Read raw data from raw_data_path and store them in self._raw_data_df.

        Args:
            raw_data_path (str): the path of raw data jsonl file
        """
        p = Path(raw_data_path)
        if not p.is_file():
            logger.error('DataDiagnosis: invalid raw data path - {}'.format(raw_data_path))
            return
        try:
            with p.open(encoding='utf-8') as f:
                for single_node_summary in jsonlines.Reader(f):
                    self._raw_data_df = self._raw_data_df.append(single_node_summary, ignore_index=True)
            self._raw_data_df = self._raw_data_df.rename(self._raw_data_df['node'])
            self._raw_data_df = self._raw_data_df.drop(columns=['node'])
        except Exception as e:
            self._raw_data_df = None
            logger.error('DataDiagnosis: invalid raw data fomat - {}'.format(str(e)))

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

    def _read_baseline(self, baseline_file=None):
        """Read baseline from baseline yaml file.

        Args:
            baseline_file (str, optional): The path of baseline yaml file. Defaults to None.

        Returns:
            dict: dict object read from yaml file
        """
        default_rule_file = Path(__file__).parent / 'default_rule.yaml'
        p = Path(baseline_file) if baseline_file else default_rule_file
        if not p.is_file():
            logger.error('DataDiagnosis: invalid rule file path - {}'.format(str(p.resolve())))
            return None
        baseline = None
        with p.open() as f:
            baseline = yaml.load(f, Loader=yaml.SafeLoader)
        return baseline

    def _set_default_criteria(self, baseline, metric):
        """Set the default criteria and rule for the metric if not set.

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
        # use variance with -0.05 as the dedault rule if not set
        if 'rules' not in baseline or (
            baseline['rules']['name'] == 'variance' and 'condition' not in baseline['rules']
        ):
            baseline['rules'] = {'name': 'variance', 'condition': -0.05}
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
        self._sb_baseline = self._read_baseline(baseline_file)
        if not self._sb_baseline:
            return False
        full_baseline = {}
        try:
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
                            full_baseline[metric] = self._set_default_criteria(full_baseline[metric], metric)
                            break
        except Exception as e:
            logger.error('DataDiagnosis: invalid rule file fomat - {}'.format(str(e)))
            return False

        self._sb_baseline = full_baseline
        return True

    def hw_issue(self, benchmark_list):
        """Idendify if the benchmark is classified as hardware issue.

        All benchmarks except models are classified hardware issue.

        Args:
            benchmark_list (list): list of benchmarks

        Returns:
            bool: return true if it's hardware issue
        """
        if not isinstance(benchmark_list, set):
            return False
        for category in benchmark_list:
            if 'models' not in category:
                return True
        return False

    def single_node_diagnosis(self, node):
        """Use rules to diagnosis single node data.

        Use the rules defined in rule_file to diagnose the raw data of each node,
        if the node violate any rule, label as issued node and save
        the 'Hw Issues', '# of Issues', 'Category', 'Issue Details' and processed data of issued node.

        Args:
            node (str): the node to do the diagosis

        Returns:
            None if the node is not labeled as issued,
            otherwise return details for the issued node.
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
                rule_op = RuleOp.get_rule_func(RuleType(rule['name']))
                (pass_rule, processed) = rule_op(data, baseline, rule)
                summary_data_row[metric] = processed
            # label the node as issued one
            if not pass_rule:
                # use return code to identify 'miss test'
                if 'return_code' in metric:
                    category_details.append(benchmark + '_miss')
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
            return (details_row, summary_data_row)

        return (None, None)

    def rule_based_diagnosis(self, rule_file):
        """Rule-based data diagnosis for multi nodes' raw data.

        Use the rules defined in rule_file to diagnose the raw data of each node,
        if the node violate any rule, label as issued node and save
        the 'Hw Issues', '# of Issues', 'Category', 'Issue Details' and processed data of issued node.

        Args:
            rule_file (str): The path of baseline yaml file

        Returns:
            DataFrame: issued nodes's detailed information
        """
        if len(self._raw_data_df) == 0:
            return False
        if not self._get_criteria(rule_file):
            return False
        summary_columns = ['Hw Issues', '# of Issues', 'Category', 'Issue Details']
        data_not_accept_df = pd.DataFrame(columns=summary_columns)
        summary_details_df = pd.DataFrame()

        for node in self._raw_data_df.index:
            (details_row, summary_data_row) = self.single_node_diagnosis(node)
            if details_row:
                data_not_accept_df.loc[node] = details_row
                summary_details_df = summary_details_df.append(summary_data_row)

        data_not_accept_df = data_not_accept_df.join(summary_details_df)
        data_not_accept_df = data_not_accept_df.sort_values(by=summary_columns, ascending=False)
        return data_not_accept_df

    def _excel_output(self, data_not_accept_df, output_file):
        """Output the processed results into excel file.

        Args:
            data_not_accept_df (DataFrame): issued nodes's detailed information
            output_file (str): the path of output excel file
        """
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        # Check whether writer is valiad
        if not isinstance(writer, pd.ExcelWriter):
            return
        self._excel_raw_data_output(writer, self._raw_data_df)
        self._excel_data_not_accept_output(writer, data_not_accept_df)
        writer.save()

    def _excel_raw_data_output(self, writer, raw_data_df):
        """Output raw data into 'Raw Data' excel page."""
        # Output the raw data
        if isinstance(raw_data_df, pd.DataFrame) and not raw_data_df.empty:
            raw_data_df.to_excel(writer, 'Raw Data', index=True)
        else:
            logger.warning('DataDiagnosis: excel_data_output - raw_data_df is empty.')

    def _excel_data_not_accept_output(self, writer, data_not_accept_df):
        """Output data_not_accept_df into 'Not Accept' excel page."""
        # Get the xlsxwriter workbook objects and init the color format
        workbook = writer.book
        # Add a format. red fill with dark red text.
        color_format_red = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        # Add a format. Light red fill with red text.
        color_format_light_red = workbook.add_format({'bg_color': '#ffebed', 'font_color': '#9e292e'})
        percent_format = workbook.add_format({'num_format': '0.00%'})

        # Output the not accept
        if isinstance(data_not_accept_df, pd.DataFrame):
            data_not_accept_df.to_excel(writer, 'Not Accept', index=True)
            if not data_not_accept_df.empty:
                row_start = 1
                row_end = max(row_start, len(data_not_accept_df))
                fix_table_len = 4
                columns = data_not_accept_df.columns
                columns = columns[fix_table_len:]
                col_start = fix_table_len
                # Get the xlsxwriter worksheet objects.
                worksheet = writer.sheets['Not Accept']

                for colums in columns:
                    col_start += 1
                    if self._sb_baseline[colums]['rules']['name'] == 'variance':
                        worksheet.conditional_format(
                            row_start,
                            col_start,
                            row_end,
                            col_start,    # start_row, start_col, end_row, end_col
                            {
                                'type': 'no_blanks',
                                'format': percent_format
                            }
                        )    # Apply percent format for the columns whose rules are variance type.
                        worksheet.conditional_format(
                            row_start,
                            col_start,
                            row_end,
                            col_start,    # start_row, start_col, end_row, end_col
                            {
                                'type': 'cell',
                                'criteria': '<=',
                                'value': self._sb_baseline[colums]['rules']['condition'],
                                'format': color_format_red
                            }
                        )    # Apply red format if the variance violates the rule.
                        worksheet.conditional_format(
                            row_start,
                            1,
                            row_end,
                            len(data_not_accept_df.columns),    # start_row, start_col, end_row, end_col
                            {
                                'type': 'cell',
                                'criteria': '<=',
                                'value': -0.03,
                                'format': color_format_light_red
                            }
                        )    # Apply light red format if the variance is lower than -3%.

        else:
            logger.warning('DataDiagnosis: excel_data_output - data_not_accept_df is empty.')
