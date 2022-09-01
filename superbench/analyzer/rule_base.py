# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A base module for rule-related module."""

import re

from superbench.common.utils import logger
from superbench.analyzer import file_handler


class RuleBase():
    """RuleBase class."""
    def __init__(self):
        """Init function."""
        self._sb_rules = {}
        self._benchmark_metrics_dict = {}
        self._enable_metrics = set()

    def _get_metrics_by_benchmarks(self, metrics_list):
        """Get mappings of benchmarks:metrics from metrics_list.

        Args:
            metrics_list (list): list of metrics

        Returns:
            dict: metrics organized by benchmarks
        """
        benchmarks_metrics = {}
        for metric in metrics_list:
            if '/' not in metric:
                logger.warning('RuleBase: get_metrics_by_benchmarks - {} does not have benchmark_name'.format(metric))
            else:
                benchmark = metric.split('/')[0]
                # support annotations in benchmark naming
                if ':' in benchmark:
                    benchmark = metric.split(':')[0]
                if benchmark not in benchmarks_metrics:
                    benchmarks_metrics[benchmark] = set()
                benchmarks_metrics[benchmark].add(metric)
        return benchmarks_metrics

    def _check_and_format_rules(self, rule, name):
        """Check the rule of the metric whether the format is valid.

        Args:
            rule (dict): the rule
            name (str): the rule name

        Returns:
            dict: the rule for the metric
        """
        # check if rule is supported
        if 'categories' not in rule:
            logger.log_and_raise(exception=Exception, msg='{} lack of category'.format(name))
        if 'metrics' in rule:
            if isinstance(rule['metrics'], str):
                rule['metrics'] = [rule['metrics']]
        return rule

    def _get_metrics(self, rule, benchmark_rules):
        """Get metrics in the rule.

        Parse metric regex in the rule, and store the (metric, -1) pair
        in _sb_rules[rule]['metrics']

        Args:
            rule (str): the name of the rule
            benchmark_rules (dict): the dict of rules
        """
        metrics_in_rule = benchmark_rules[rule]['metrics']
        benchmark_metrics_dict_in_rule = self._get_metrics_by_benchmarks(metrics_in_rule)
        for benchmark_name in benchmark_metrics_dict_in_rule:
            if benchmark_name not in self._benchmark_metrics_dict:
                logger.warning('RuleBase: get metrics failed - {}'.format(benchmark_name))
                continue
            # get rules and criteria for each metric
            for metric in self._benchmark_metrics_dict[benchmark_name]:
                # metric full name in baseline
                if metric in metrics_in_rule:
                    self._sb_rules[rule]['metrics'][metric] = -1
                    self._enable_metrics.add(metric)
                    continue
                # metric full name not in baseline, use regex to match
                for metric_regex in benchmark_metrics_dict_in_rule[benchmark_name]:
                    if re.search(metric_regex, metric):
                        self._sb_rules[rule]['metrics'][metric] = -1
                        self._enable_metrics.add(metric)

    def _preprocess(self, raw_data_file, rule_file):
        """Preprocess/preparation operations for the rules.

        Args:
            raw_data_file (str): the path of raw data file
            rule_file (str): the path of rule file

        Returns:
            dict: dict of rules
        """
        # read raw data from file
        self._raw_data_df = file_handler.read_raw_data(raw_data_file)
        # re-organize metrics by benchmark names
        self._benchmark_metrics_dict = self._get_metrics_by_benchmarks(list(self._raw_data_df.columns))
        # check raw data whether empty
        if len(self._raw_data_df) == 0:
            logger.log_and_raise(exception=Exception, msg='RuleBase: empty raw data')
        # read rules
        rules = file_handler.read_rules(rule_file)
        return rules
