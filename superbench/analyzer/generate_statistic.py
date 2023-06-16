# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for baseline generation."""

import argparse
import os
import natsort as ns

from joblib import Parallel, delayed
import pandas as pd
import matplotlib.pyplot as plt

from superbench.common.utils import logger
from superbench.analyzer import file_handler
from superbench.analyzer import data_analysis
from superbench.analyzer.diagnosis_rule_op import RuleOp, DiagnosisRuleType
from generate_baseline import GenerateBaseline


def plot_steps(data, title=None, save_path=None, show=True):
    """Plot steps.

    Args:
        data (list): data to plot
        title (str): title of the plot
        save_path (str): path to save the plot
        show (bool): whether to show the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(0, len(data)), data)
    if title:
        plt.title(title)
    plt.xlabel('Devices')
    plt.ylabel('Value')
    plt.ylim(0, max(data) * 1.1)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


class GenerateStatistics(GenerateBaseline):
    """GenerateStatistics class to generate statistics for raw data."""
    def calculate_statistics(self, healthy_df):
        """Calculate statistics for healthy data.

        Args:
            healthy_df (DataFrame): healthy data

        Returns:
            DataFrame: statistics for healthy data
        """
        stat_df = data_analysis.statistic(healthy_df)
        stat_df.loc['(max-min)/max'] = (stat_df.loc['max'] - stat_df.loc['min']) / stat_df.loc['max']
        stat_df = stat_df.drop(index='1%')
        stat_df = stat_df.drop(index='5%')
        stat_df = stat_df.drop(index='95%')
        stat_df = stat_df.drop(index='99%')
        return stat_df

    def output_excel(self, excel_file, stat_df, digit=2):
        """Output excel file.

        Args:
            excel_file (str): excel file path
            stat_df (DataFrame): statistics data
            digit (int): digit to round
        """
        try:
            writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

            for benchmark in self._benchmark_metrics_dict:
                benchmark_df = stat_df[self._benchmark_metrics_dict[benchmark]]
                #benchmark_df = benchmark_df[,mixedsort(names(benchmark_df))]
                benchmark_df = benchmark_df.reindex(ns.natsorted(benchmark_df.columns), axis=1)
                sheet_name = benchmark if len(benchmark) <= 30 else benchmark.split('-')[-1]
                benchmark_df.to_excel(writer, sheet_name=sheet_name)
                worksheet = writer.sheets[sheet_name]
                row_start = 1
                row_end = max(row_start, len(self._benchmark_metrics_dict[benchmark]))
                columns = list(benchmark_df.index)
                col_index = columns.index('(max-min)/max') + 1
                workbook = writer.book
                percent_format = workbook.add_format({'num_format': '0.00%'})
                worksheet.conditional_format(
                    col_index,
                    row_start,
                    col_index,
                    row_end,    # start_row, start_col, end_row, end_col
                    {
                        'type': 'no_blanks',
                        'format': percent_format
                    }
                )
                num_format = f'0.{digit * "0"}'
                for col_index in range(2, len(columns)):
                    round_format = workbook.add_format({'num_format': num_format})
                    worksheet.conditional_format(
                        col_index,
                        row_start,
                        col_index,
                        row_end,    # start_row, start_col, end_row, end_col
                        {
                            'type': 'no_blanks',
                            'format': round_format
                        }
                    )
            writer.close()
        except Exception as e:
            logger.error('output excel failed: {}'.format(str(e)))

    def run(self, raw_data_file, output_dir, diagnosis_rule_file=None, summary_rule_file=None, digit=2, plot=False):
        """Run the statistics generation.

        Args:
            raw_data_file (str): raw data file path
            output_dir (str): output directory
            diagnosis_rule_file (str): diagnosis rule file path
            summary_rule_file (str): summary rule file path
            digit (int): digit to round
            plot (bool): whether to plot the data
        """
        try:
            # aggregate results from different devices
            self._raw_data_df = self.get_aggregate_data(raw_data_file, summary_rule_file)
            # re-organize metrics by benchmark names
            self._benchmark_metrics_dict = self._get_metrics_by_benchmarks(list(self._raw_data_df.columns))
            # read existing baseline
            baseline = {}
            # read diagnosis rules
            aggregated_df = self._raw_data_df.copy()
            rules = file_handler.read_rules(diagnosis_rule_file)
            if not self._parse_rules_and_baseline(rules, baseline):
                logger.error('parse rule failed')
                return None
            else:
                for rule in self._sb_rules:
                    single_metric_rule = self._sb_rules[rule]
                    metrics = list(single_metric_rule['metrics'].keys())
                    function_name = self._sb_rules[rule]['function']
                    rule_op = RuleOp.get_rule_func(DiagnosisRuleType(function_name))
                    outputs = Parallel(n_jobs=-1)(
                        delayed(self.fix_threshold_outlier_detection)
                        (aggregated_df[metric], single_metric_rule, metric, rule_op) for metric in metrics
                    )
                    for index, out in enumerate(outputs):
                        if not out:
                            logger.error('Analyzer: filter healthy nodese failed')
                            return
                        aggregated_df[metrics[index]] = out[1]
                        if plot:
                            plot_steps(
                                out[1].tolist(),
                                title=metrics[index],
                                save_path=os.path.join(
                                    output_dir, 'figures', metrics[index].replace('/', '_').replace(':', '_') + '.png'
                                ),
                                show=False
                            )
            stat_df = self.calculate_statistics(aggregated_df)
            excel_file = os.path.join(output_dir, 'benchmark_stability_stat.xlsx')
            self.output_excel(excel_file, stat_df, digit)

        except Exception as e:
            logger.error('Analyzer: generate statisitics failed, msg: {}'.format(str(e)))


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir',
        type=str,
        default='rawdata/',
        required=False,
        help='Input directory which stores the results-summary.jsonl.'
    )
    parser.add_argument(
        '--diagnosis_rule_file',
        type=str,
        default='rules/diagnosis_rules.yaml',
        required=False,
        help='The input path of diagnosis rule file.'
    )
    parser.add_argument(
        '--summary_rule_file',
        type=str,
        default='rules/analysis_rules.yaml',
        required=False,
        help='The input path of summary rule file.'
    )
    args = parser.parse_args()

    # use fix threshold method, need result_summary rules to define how to aggregate the metrics and diagnosis_rules.yaml to define the rules for the metrics.
    GenerateStatistics().run(
        args.input_dir + '/results-summary.jsonl', args.input_dir, args.diagnosis_rule_file, args.summary_rule_file
    )
