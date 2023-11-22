# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for baseline generation."""

from copy import deepcopy
from pathlib import Path
import json
import re

from joblib import Parallel, delayed
import pandas as pd

from superbench.common.utils import logger
from superbench.analyzer import file_handler
from superbench.analyzer import data_analysis
from superbench.analyzer import DataDiagnosis
from superbench.analyzer import ResultSummary
from superbench.analyzer.diagnosis_rule_op import RuleOp, DiagnosisRuleType


class BaselineGeneration(DataDiagnosis):
    """The class to generate baseline for raw data."""
    def fix_threshold_outlier_detection(self, data_series, single_metric_with_baseline, metric, rule_op):
        """Fix threshold outlier detection algorithm.

        Step 0: Put all data in the collection.
        Step 1: Regenerate the collection.
          Calculate the average number in the collection as the baseline.
          Remove all data which cannot pass the fix threshold based on the new baseline.
        Step 2: If no data has been removed from Step 1, go to Step 3; otherwise, go to Step 1.
        Step 3: Use the baseline and fix threshold for Outlier Detection.

        Args:
            data_series (pd.Series): data of the metric.
            single_metric_with_baseline (dict): baseline of the single metric in 'metrics' in 2-layer dict format.
            metric (str): the name of the metric to execute the algorithm.
            rule_op (function): diagnosis rule op function.

        Returns:
            tuple: the baseline of the metric, normal data of the metric.
        """
        if single_metric_with_baseline['metrics'][metric] \
                is not None and single_metric_with_baseline['metrics'][metric] != -1:
            return single_metric_with_baseline['metrics'][metric]
        tmp_single_metric_with_baseline = deepcopy(single_metric_with_baseline)
        tmp_single_metric_with_baseline['metrics'] = {}
        clean = False
        while clean is False:
            clean = True
            baseline_val = data_series.mean()
            for val in data_series.index:
                tmp_single_metric_with_baseline['metrics'][metric] = baseline_val
                if baseline_val == 0:
                    break
                data_row = pd.Series([data_series[val]], index=[metric])
                details = []
                categories = set()
                summary_data_row = pd.Series(index=[metric], dtype=float)
                violated_num = rule_op(data_row, tmp_single_metric_with_baseline, summary_data_row, details, categories)
                if violated_num:
                    data_series = data_series.drop(val)
                    clean = False
        baseline = tmp_single_metric_with_baseline['metrics'][metric]
        return baseline, data_series

    def get_aggregate_data(self, raw_data_file, summary_rule_file):
        r"""Aggregate raw data according to the summary rule file.

        If the metric is aggregated by rank (:\d+), remove the rank info to generate the metric name and aggregate data.
        If the metric is aggregated by regex pattern, aggregate the data and copy to all metrics matches this pattern.

        Args:
            raw_data_file (str): the file name of the raw data file.
            summary_rule_file (str): the file name of the summary rule file.

        Returns:
            DataFrame: aggregated data
        """
        self.rs = ResultSummary()
        rules = self.rs._preprocess(raw_data_file, summary_rule_file)
        # parse rules for result summary
        if not self.rs._parse_rules(rules):
            return
        aggregated_df = pd.DataFrame()
        for rule in self.rs._sb_rules:
            single_metric_rule = self.rs._sb_rules[rule]
            metrics = list(single_metric_rule['metrics'].keys())
            data_df_of_rule = self.rs._raw_data_df[metrics]
            if self.rs._sb_rules[rule]['aggregate']:
                # if aggregate is True, aggregate in ranks
                if self.rs._sb_rules[rule]['aggregate'] is True:
                    data_df_of_rule = data_analysis.aggregate(data_df_of_rule)
                # if aggregate is not empty and is a pattern in regex, aggregate according to pattern
                else:
                    pattern = self.rs._sb_rules[rule]['aggregate']
                    data_df_of_rule_with_short_name = data_analysis.aggregate(data_df_of_rule, pattern)
                    data_df_of_rule = pd.DataFrame(columns=metrics)
                    # restore the columns of data_fd to full metric names
                    for metric in metrics:
                        short = ''
                        match = re.search(pattern, metric)
                        if match:
                            metric_in_list = list(metric)
                            for i in range(1, len(match.groups()) + 1):
                                metric_in_list[match.start(i):match.end(i)] = '*'
                            short = ''.join(metric_in_list)
                        data_df_of_rule[metric] = data_df_of_rule_with_short_name[short]
            aggregated_df = pd.concat([aggregated_df, data_df_of_rule], axis=1)
        return aggregated_df

    def generate_baseline(self, algo, aggregated_df, diagnosis_rule_file, baseline):
        """Generate the baseline in json format.

        Args:
            algo (str): the algorithm to generate the baseline.
            aggregated_df (DataFrame): aggregated data.
            diagnosis_rule_file (str): the file name of the diagnosis rules which used in fix_threshold algorithm.
            baseline (dict): existing baseline of some metrics.

        Returns:
            dict: baseline of metrics defined in diagnosis_rule_files for fix_threshold algorithm or
                  defined in rule_summary_files for mean.
        """
        # re-organize metrics by benchmark names
        self._benchmark_metrics_dict = self._get_metrics_by_benchmarks(list(self._raw_data_df.columns))
        if algo == 'mean':
            mean_df = self._raw_data_df.mean()
            for metric in self._raw_data_df.columns:
                if metric in baseline:
                    return baseline[metric]
                baseline[metric] = mean_df[metric]
        elif algo == 'fix_threshold':
            # read diagnosis rules
            rules = file_handler.read_rules(diagnosis_rule_file)
            if not self._parse_rules_and_baseline(rules, baseline):
                return baseline
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
                        baseline[metrics[index]] = out[0]
                        aggregated_df[metrics[index]] = out[1]
        return baseline

    def run(
        self, raw_data_file, summary_rule_file, diagnosis_rule_file, pre_baseline_file, algorithm, output_dir, digit=2
    ):
        """Export baseline to json file.

        Args:
            raw_data_file (str): Path to raw data jsonl file.
            summary_rule_file (str): the file name of the summary rule file.
            diagnosis_rule_file (str): the file name of the diagnosis rules which used in fix_threshold algorithm.
            pre_baseline_file (str): the file name of the previous baseline file.
            algorithm (str): the algorithm to generate the baseline.
            output_dir (str): the directory to save the baseline file.
            digit (int): the number of digits after the decimal point.
        """
        try:
            # aggregate results from different devices
            self._raw_data_df = self.get_aggregate_data(raw_data_file, summary_rule_file)
            # read existing baseline
            baseline = {}
            if pre_baseline_file:
                baseline = file_handler.read_baseline(pre_baseline_file)
            # generate baseline accordint to rules in diagnosis and fix threshold outlier detection method
            baseline = self.generate_baseline(algorithm, self._raw_data_df, diagnosis_rule_file, baseline)
            for metric in baseline:
                val = baseline[metric]
                if metric in self._raw_data_df:
                    if isinstance(self._raw_data_df[metric].iloc[0], float):
                        baseline[metric] = f'%.{digit}g' % val if abs(val) < 1 else f'%.{digit}f' % val
                    elif isinstance(self._raw_data_df[metric].iloc[0], int):
                        baseline[metric] = int(val)
                    else:
                        try:
                            baseline[metric] = float(val)
                        except Exception as e:
                            logger.error('Analyzer: {} baseline is not numeric, msg: {}'.format(metric, str(e)))
            baseline = json.dumps(baseline, indent=2, sort_keys=True)
            baseline = re.sub(r': \"(\d+.?\d*)\"', r': \1', baseline)
            with (Path(output_dir) / 'baseline.json').open('w') as f:
                f.write(baseline)

        except Exception as e:
            logger.error('Analyzer: generate baseline failed, msg: {}'.format(str(e)))
