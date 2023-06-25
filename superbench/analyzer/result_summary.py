# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for Result Summary."""

import re
from pathlib import Path

import pandas as pd

from superbench.common.utils import logger
from superbench.analyzer import file_handler
from superbench.analyzer.summary_op import SummaryOp, SummaryType
from superbench.analyzer import RuleBase
from superbench.analyzer import data_analysis


class ResultSummary(RuleBase):
    """Result summary class."""
    def _check_rules(self, rule, name):
        """Check whether the formart of the rule is valid.

        Args:
            rule (dict): the rule
            name (str): the rule name

        Returns:
            dict: the rule for the metric
        """
        # check if rule is supported
        super()._check_and_format_rules(rule, name)
        if 'metrics' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of metrics'.format(name))
        if 'statistics' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of function'.format(name))
        # convert single statistic str to list
        if not isinstance(rule['statistics'], list):
            rule['statistics'] = [rule['statistics']]
        # check statistics format, should be SummaryType or p\d\d?
        for function in rule['statistics']:
            try:
                if not (re.fullmatch(r'p\d\d?', function) or isinstance(SummaryType(function), SummaryType)):
                    logger.log_and_raise(
                        exception=Exception, msg='{} has invalid statistics name {}'.format(name, function)
                    )
            except Exception:
                logger.log_and_raise(
                    exception=Exception, msg='{} has invalid statistics name {}'.format(name, function)
                )
        # check aggregate format, should be None or bool or pattern in regex with () group
        if 'aggregate' in rule and not isinstance(rule['aggregate'],
                                                  bool) and not re.search(r'\(.*\)', rule['aggregate']):
            logger.log_and_raise(exception=Exception, msg='{} aggregate must be bool type'.format(name))
        return rule

    def _parse_rules(self, rules):
        """Parse the rules for result summary.

        Args:
            rules (dict): rules from rule yaml file

        Returns:
            bool: return True if successfully get all rules, otherwise False.
        """
        try:
            if not rules:
                logger.error('ResultSummary: get rules failed')
                return False
            self._sb_rules = {}
            self._enable_metrics = set()
            benchmark_rules = rules['superbench']['rules']
            for rule in benchmark_rules:
                benchmark_rules[rule] = self._check_rules(benchmark_rules[rule], rule)
                self._sb_rules[rule] = {}
                self._sb_rules[rule]['name'] = rule
                self._sb_rules[rule]['categories'] = benchmark_rules[rule]['categories']
                self._sb_rules[rule]['metrics'] = {}
                self._sb_rules[rule]['statistics'] = benchmark_rules[rule]['statistics']
                self._sb_rules[rule][
                    'aggregate'] = benchmark_rules[rule]['aggregate'] if 'aggregate' in benchmark_rules[rule] else False
                super()._get_metrics(rule, benchmark_rules)
            return True
        except Exception as e:
            logger.error('ResultSummary: parse rules failed - {}'.format(str(e)))
            return False

    def _format_summary_of_rule(self, category, summary_df_of_rule, statistics):
        """Format summary_df of a rule info list of lines.

        Args:
            category (str): category in the rule
            summary_df_of_rule (DataFrame): summary df of a rule, the columns are metrics, the index are statistics
            statistics (list): statistics in the rule

        Returns:
            list: list of summary lines like [category, metric, statistic, value]
        """
        summary = []
        metrics = summary_df_of_rule.columns
        if metrics.empty is True:
            for statistic in statistics:
                summary.append([category, '', statistic, ''])
        for metric in metrics:
            for statistic in statistics:
                summary.append([category, metric, statistic, summary_df_of_rule.loc[statistic, metric]])
        return summary

    def _merge_summary(self, summary):
        """Merge summary of multiple rules into DataFrame.

        Args:
            summary (dict): summary dict, the keys are categories, the values are summary lines for the category

        Returns:
            DataFrame: summary of all rules
        """
        summary_df = pd.DataFrame()
        for category in summary:
            for i in range(len(summary[category])):
                summary_df = pd.concat([summary_df, pd.DataFrame([summary[category][i]])], ignore_index=True)
        return summary_df

    def _generate_summary(self, round):
        r"""Generate summay dict of all rules.

        For each rule, aggregate the data by user-defined pattern or ranks (:\\d+), calculate
        the list of statistics of aggregated metrics, then format the summary in {category, lines}.

        Args:
            round (int): the number of decimal digits

        Returns:
            dict: summary dict, the keys are categories, the values are summary lines for the category
        """
        summary = {}
        for rule in self._sb_rules:
            metrics = list(self._sb_rules[rule]['metrics'].keys())
            category = self._sb_rules[rule]['categories']
            data_df_of_rule = self._raw_data_df[metrics]
            statistics = self._sb_rules[rule]['statistics']
            summary_df_of_rule = pd.DataFrame()
            # skip metrics aggregation and statistics calculation fot the rule with no matched metrics
            if len(metrics) != 0:
                if self._sb_rules[rule]['aggregate']:
                    # if aggregate is True, aggregate in ranks
                    if self._sb_rules[rule]['aggregate'] is True:
                        data_df_of_rule = data_analysis.aggregate(data_df_of_rule)
                    # if aggregate is not empty and is a pattern in regex, aggregate according to pattern
                    else:
                        data_df_of_rule = data_analysis.aggregate(data_df_of_rule, self._sb_rules[rule]['aggregate'])
                summary_df_of_rule = pd.DataFrame(columns=sorted(data_df_of_rule.columns))
                for statistic_name in statistics:
                    # get SummaryOp and calculate statistics
                    # if statistic_name is 'p\d\d?', SummaryOp should be pencentile
                    if str.startswith(statistic_name, 'p'):
                        rule_op = SummaryOp.get_summary_func(SummaryType('percentile'))
                        val = int(statistic_name.strip('p'))
                        summary_df_of_rule.loc[statistic_name] = rule_op(data_df_of_rule, val)
                    else:
                        rule_op = SummaryOp.get_summary_func(SummaryType(statistic_name))
                        summary_df_of_rule.loc[statistic_name] = rule_op(data_df_of_rule)
                # format values to n significant decimal digits
                if round and isinstance(round, int):
                    summary_df_of_rule = data_analysis.round_significant_decimal_places(
                        summary_df_of_rule, round, list(summary_df_of_rule.columns)
                    )
            # format summary_df of a rule to list of lines
            summary_lines_of_rule = self._format_summary_of_rule(category, summary_df_of_rule, statistics)
            summary[category] = summary_lines_of_rule

        return summary

    def generate_md_lines(self, summary):
        """Generate text in markdown foramt.

        Use category to be the 2nd-header, use tables to show the data

        Args:
            summary (dict): summary dict, the keys are categories, the values are summary lines for the category

        Returns:
            list: lines in markdown format
        """
        lines = []
        for category in summary:
            lines.append('## {}\n'.format(category))
            summary_df = pd.DataFrame(summary[category])
            summary_df = summary_df.drop(columns=0, axis=1)
            header = ['metric', 'statistics', 'values']
            table_lines = file_handler.generate_md_table(summary_df, header)
            lines.extend(table_lines)
            lines.append('\n')
        return lines

    def output_summary_in_excel(self, raw_data_df, summary, output_path):
        """Output result summary in excel foramt.

        Args:
            raw_data_df (DataFrame): the DataFrame of raw data df
            summary (DataFrame): the DataFrame of summary
            output_path (str): the path of output file
        """
        try:
            writer = pd.ExcelWriter(output_path, engine='openpyxl')
            # check whether writer is valiad
            if not isinstance(writer, pd.ExcelWriter):
                logger.error('ResultSummary: excel_data_output - invalid file path.')
                return
            # output the raw data in 'Raw Data' sheet
            file_handler.output_excel_raw_data(writer, raw_data_df, 'Raw Data')
            # output the result summary in 'Summary' sheet
            if isinstance(summary, pd.DataFrame) and not summary.empty:
                summary.to_excel(writer, 'Summary', index=False, header=False)
                worksheet = writer.sheets['Summary']
                row = worksheet.max_row
                # merge cells in 'category' column with the same category
                file_handler.merge_column_in_excel(worksheet, row, 1)
            else:
                logger.error('ResultSummary: excel_data_output - summary is empty.')
            writer.close()
        except Exception as e:
            logger.error('ResultSummary: excel_data_output - {}'.format(str(e)))

    def run(self, raw_data_file, rule_file, output_dir, output_format, round=2):
        """Run the main process of result summary.

        Args:
            raw_data_file (str): the path of raw data jsonl file.
            rule_file (str): The path of baseline yaml file
            output_dir (str): the directory of output file
            output_format (str): the format of the output, 'excel' or 'md' or 'html'
            round (int): the number of decimal digits
        """
        try:
            rules = self._preprocess(raw_data_file, rule_file)
            # parse rules for result summary
            if not self._parse_rules(rules):
                return
            # generate result summary for each category
            summary = self._generate_summary(round)
            # output result summary to file
            output_path = ''
            if output_format == 'excel':
                output_path = str(Path(output_dir) / 'results-summary.xlsx')
                summary_df = self._merge_summary(summary)
                self.output_summary_in_excel(self._raw_data_df, summary_df, output_path)
            elif output_format == 'md':
                output_path = str(Path(output_dir) / 'results-summary.md')
                lines = self.generate_md_lines(summary)
                file_handler.output_lines_in_md(lines, output_path)
            elif output_format == 'html':
                output_path = str(Path(output_dir) / 'results-summary.html')
                lines = self.generate_md_lines(summary)
                file_handler.output_lines_in_html(lines, output_path)
            else:
                logger.error('ResultSummary: output failed - unsupported output format')
            logger.info('ResultSummary: Output results to {}'.format(output_path))
        except Exception as e:
            logger.error('ResultSummary: run failed - {}'.format(str(e)))
