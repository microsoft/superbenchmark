# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        data_statistics_df = raw_data_df.describe()
        data_statistics_df.loc['1%'] = raw_data_df.quantile(0.01)
        data_statistics_df.loc['5%'] = raw_data_df.quantile(0.05)
        data_statistics_df.loc['95%'] = raw_data_df.quantile(0.95)
        data_statistics_df.loc['99%'] = raw_data_df.quantile(0.99)
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
