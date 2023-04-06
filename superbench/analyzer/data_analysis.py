# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

from superbench.common.utils import logger


def statistic(raw_data_df):
    """Get the statistics of the raw data.

    The statistics include count, mean, std, min, max, 1%, 5%, 25%, 50%, 75%, 95%, 99%.

    Args:
        raw_data_df (DataFrame): raw data

    Returns:
        DataFrame: data statistics
    """
    data_statistics_df = pd.DataFrame()
    if not isinstance(raw_data_df, pd.DataFrame):
        logger.error('DataAnalyzer: the type of raw data is not pd.DataFrame')
        return data_statistics_df
    if len(raw_data_df) == 0:
        logger.warning('DataAnalyzer: empty data.')
        return data_statistics_df
    try:
        raw_data_df = raw_data_df.apply(pd.to_numeric, errors='coerce')
        raw_data_df = raw_data_df.dropna(axis=1, how='all')
        data_statistics_df = raw_data_df.describe()
        data_statistics_df.loc['1%'] = raw_data_df.quantile(0.01, numeric_only=True)
        data_statistics_df.loc['5%'] = raw_data_df.quantile(0.05, numeric_only=True)
        data_statistics_df.loc['95%'] = raw_data_df.quantile(0.95, numeric_only=True)
        data_statistics_df.loc['99%'] = raw_data_df.quantile(0.99, numeric_only=True)
        statistics_error = []
        for column in list(raw_data_df.columns):
            if column not in list(data_statistics_df.columns) and not raw_data_df[column].isnull().all():
                statistics_error.append(column)
        if statistics_error:
            logger.warning(
                'DataAnalyzer: [{}] is missing in statistics results.'.format(
                    ','.join(str(x) for x in statistics_error)
                )
            )
    except Exception as e:
        logger.error('DataAnalyzer: statistic failed, msg: {}'.format(str(e)))
    return data_statistics_df


def interquartile_range(raw_data_df):
    """Get outlier detection bounds using IQR method.

     The reference of IQR is https://en.wikipedia.org/wiki/Interquartile_range.
     Get the mild and extreme outlier upper and lower value and bound.
     values:
        Mild Outlier: A point beyond inner whiskers on either side
            lower whisker: Q1 - 1.5*IQR
            upper whisker : Q3 + 1.5*IQR
        Extreme Outlier: A point beyond outer whiskers on either side
            lower whisker : Q1 - 3*IQR
            upper whisker : Q3 + 3*IQR
     bounds:
        (values - mean) / mean

    Args:
        raw_data_df (DataFrame): raw data

    Returns:
        DataFrame: data statistics and IQR bound
    """
    if not isinstance(raw_data_df, pd.DataFrame):
        logger.error('DataAnalyzer: the type of raw data is not pd.DataFrame')
        return pd.DataFrame()
    if len(raw_data_df) == 0:
        logger.warning('DataAnalyzer: empty data.')
        return pd.DataFrame()
    try:
        data_statistics_df = statistic(raw_data_df)
        data_statistics_df.loc['mild_outlier_upper'] = data_statistics_df.loc[
            '75%'] + 1.5 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
        data_statistics_df.loc['extreme_outlier_upper'] = data_statistics_df.loc[
            '75%'] + 3 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
        data_statistics_df.loc['mild_outlier_lower'] = data_statistics_df.loc[
            '25%'] - 1.5 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
        data_statistics_df.loc['extreme_outlier_lower'] = data_statistics_df.loc[
            '25%'] - 3 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
        data_statistics_df.loc['mild_outlier_upper_bound'] = (
            data_statistics_df.loc['mild_outlier_upper'] - data_statistics_df.loc['mean']
        ) / data_statistics_df.loc['mean']
        data_statistics_df.loc['extreme_outlier_upper_bound'] = (
            data_statistics_df.loc['extreme_outlier_upper'] - data_statistics_df.loc['mean']
        ) / data_statistics_df.loc['mean']
        data_statistics_df.loc['mild_outlier_lower_bound'] = (
            data_statistics_df.loc['mild_outlier_lower'] - data_statistics_df.loc['mean']
        ) / data_statistics_df.loc['mean']
        data_statistics_df.loc['extreme_outlier_lower_bound'] = (
            data_statistics_df.loc['extreme_outlier_lower'] - data_statistics_df.loc['mean']
        ) / data_statistics_df.loc['mean']
    except Exception as e:
        logger.error('DataAnalyzer: interquartile_range failed, msg: {}'.format(str(e)))
    return data_statistics_df


def correlation(raw_data_df):
    """Get the correlations.

    Args:
        raw_data_df (DataFrame): raw data

    Returns:
        DataFrame: correlations
    """
    data_corr_df = pd.DataFrame()
    if not isinstance(raw_data_df, pd.DataFrame):
        logger.error('DataAnalyzer: the type of raw data is not pd.DataFrame')
        return data_corr_df
    if len(raw_data_df) == 0:
        logger.warning('DataAnalyzer: empty data.')
        return data_corr_df
    try:
        raw_data_df = raw_data_df.apply(pd.to_numeric, errors='coerce')
        raw_data_df = raw_data_df.dropna(axis=1, how='all')
        data_corr_df = raw_data_df.corr()
        statistics_error = []
        for column in list(raw_data_df.columns):
            if column not in list(data_corr_df.columns) and not raw_data_df[column].isnull().all():
                statistics_error.append(column)
        if statistics_error:
            logger.warning(
                'DataAnalyzer: [{}] is missing in correlation results.'.format(
                    ','.join(str(x) for x in statistics_error)
                )
            )
    except Exception as e:
        logger.error('DataAnalyzer: correlation failed, msg: {}'.format(str(e)))
    return data_corr_df


def creat_boxplot(raw_data_df, columns, output_dir):
    """Plot the boxplot for selected columns.

    Args:
        raw_data_df (DataFrame): raw data
        columns (list): selected metrics to plot the boxplot
        output_dir (str): the directory of output file
    """
    if not isinstance(raw_data_df, pd.DataFrame):
        logger.error('DataAnalyzer: the type of raw data is not pd.DataFrame')
        return
    if len(raw_data_df) == 0:
        logger.error('DataAnalyzer: empty data for boxplot.')
        return
    if not isinstance(columns, list):
        logger.error('DataAnalyzer: the type of columns should be list.')
        return
    try:
        data_columns = raw_data_df.columns
        for column in columns:
            if column not in data_columns or raw_data_df[column].dtype is not np.dtype('float'):
                logger.warning('DataAnalyzer: invalid column {} for boxplot.'.format(column))
                columns.remove(column)
        n = len(columns)
        for i in range(n):
            sns.set(style='whitegrid')
            plt.subplot(n, 1, i + 1)
            sns.boxplot(x=columns[i], data=raw_data_df, orient='h')
        plt.subplots_adjust(hspace=1)
        plt.savefig(output_dir + '/boxplot.png')
        plt.show()
    except Exception as e:
        logger.error('DataAnalyzer: creat_boxplot failed, msg: {}'.format(str(e)))


def generate_baseline(raw_data_df, output_dir):
    """Export baseline to json file.

    Args:
        raw_data_df (DataFrame): raw data
        output_dir (str): the directory of output file
    """
    try:
        raw_data_df = raw_data_df.apply(pd.to_numeric, errors='coerce')
        raw_data_df = raw_data_df.dropna(axis=1, how='all')
        if not isinstance(raw_data_df, pd.DataFrame):
            logger.error('DataAnalyzer: the type of raw data is not pd.DataFrame')
            return
        if len(raw_data_df) == 0:
            logger.error('DataAnalyzer: empty data.')
            return
        mean_df = raw_data_df.mean()
        mean_df.to_json(output_dir + '/baseline.json')
    except Exception as e:
        logger.error('DataAnalyzer: generate baseline failed, msg: {}'.format(str(e)))


def round_significant_decimal_places(df, digit, cols):
    """Format the numbers in selected columns of DataFrame n significant decimal places.

    Args:
        df (DataFrame): the DataFrame to format
        digit (int): the number of decimal places
        cols (list): the selected columns

    Returns:
        DataFrame: the DataFrame after format
    """
    format_significant_str = '%.{}g'.format(digit)
    for col in cols:
        if np.issubdtype(df[col], np.number):
            df[col] = df[col].map(
                lambda x: float(format_significant_str % x) if abs(x) < 1 else round(x, digit), na_action='ignore'
            )
    return df


def aggregate(raw_data_df, pattern=None):
    r"""Aggregate data of multiple ranks or multiple devices.

    By default, aggregate results of multiple ranks like 'metric:\\d+' for most metrics.
    For example, aggregate the results of kernel-launch overhead
    from 8 GPU devices into one collection.
    If pattern is given, use pattern to match metric and replace matched part in metric to *
    to generate a aggregated metric name and then aggpregate these metrics' data.

    Args:
        raw_data_df (DataFrame): raw data

    Returns:
        DataFrame: the dataframe of aggregated data
    """
    try:
        metric_store = {}
        metrics = list(raw_data_df.columns)
        for metric in metrics:
            short = metric.strip(metric.split(':')[-1]).strip(':')
            if pattern:
                match = re.search(pattern, metric)
                if match:
                    metric_in_list = list(metric)
                    for i in range(1, len(match.groups()) + 1):
                        metric_in_list[match.start(i):match.end(i)] = '*'
                    short = ''.join(metric_in_list)
            if short not in metric_store:
                metric_store[short] = []
            metric_store[short].extend(raw_data_df[metric].tolist())
        df = pd.DataFrame()
        for short in metric_store:
            df = pd.concat([df, pd.DataFrame(metric_store[short], columns=[short])], axis=1)
        return df
    except Exception as e:
        logger.error('DataAnalyzer: aggregate failed, msg: {}'.format(str(e)))
        return None
