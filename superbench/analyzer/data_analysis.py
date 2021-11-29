# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data analysis."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from superbench.common.utils import logger


def statistic(raw_data_df):
    """Get the statistics of the raw data, including count, mean, std, min, max, 25%, 50%, 75%.

    Args:
        raw_data_df (DataFrame): raw data

    Returns:
        DataFrame: data statistics
    """
    if len(raw_data_df) == 0:
        return pd.DataFrame()
    data_statistics_df = raw_data_df.describe()
    statistics_error = []
    for column in list(raw_data_df.columns):
        if column not in list(data_statistics_df.columns) and not raw_data_df[column].isnull().all():
            statistics_error.append(column)
    if statistics_error:
        logger.warning(
            'DataAnalyzer: [{}] is missing in statistics results.'.format(','.join(str(x) for x in statistics_error))
        )
    return data_statistics_df


def inter_quartile_range(raw_data_df):
    """Get the IQR outlier detection bound, including mild and extreme outlier upper and lower value and bound.

    Args:
        raw_data_df (DataFrame): raw data

    Returns:
        DataFrame: data statistics and IQR bound
    """
    if len(raw_data_df) == 0:
        return pd.DataFrame()
    data_statistics_df = statistic(raw_data_df)
    data_statistics_df.loc['mild_outlier_upper'] = data_statistics_df.loc[
        '75%'] + 1.5 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
    data_statistics_df.loc['mild_outlier_upper_bound'] = (
        data_statistics_df.loc['mild_outlier_upper'] - data_statistics_df.loc['mean']
    ) / data_statistics_df.loc['mean']
    data_statistics_df.loc['extreme_outlier_upper'] = data_statistics_df.loc[
        '75%'] + 3 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
    data_statistics_df.loc['extreme_outlier_upper_bound'] = (
        data_statistics_df.loc['extreme_outlier_upper'] - data_statistics_df.loc['mean']
    ) / data_statistics_df.loc['mean']
    data_statistics_df.loc['mild_outlier_lower'] = data_statistics_df.loc[
        '25%'] - 1.5 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
    data_statistics_df.loc['mild_outlier_lower_bound'] = (
        data_statistics_df.loc['mild_outlier_lower'] - data_statistics_df.loc['mean']
    ) / data_statistics_df.loc['mean']
    data_statistics_df.loc['extreme_outlier_lower'] = data_statistics_df.loc[
        '25%'] - 3 * (data_statistics_df.loc['75%'] - data_statistics_df.loc['25%'])
    data_statistics_df.loc['extreme_outlier_lower_bound'] = (
        data_statistics_df.loc['extreme_outlier_lower'] - data_statistics_df.loc['mean']
    ) / data_statistics_df.loc['mean']

    return data_statistics_df


def correlation(raw_data_df):
    """Get the correlations.

    Args:
        raw_data_df (DataFrame): raw data

    Returns:
        DataFrame: correlations
    """
    if len(raw_data_df) == 0:
        return pd.DataFrame()
    data_corr_df = raw_data_df.corr()
    statistics_error = []
    for column in list(raw_data_df.columns):
        if column not in list(data_corr_df.columns) and not raw_data_df[column].isnull().all():
            statistics_error.append(column)
    if statistics_error:
        logger.warning(
            'DataAnalyzer: [{}] is missing in correlation results.'.format(','.join(str(x) for x in statistics_error))
        )
    return data_corr_df


def creat_boxplot(raw_data_df, columns):
    """Plot the boxplot for selected columns.

    Args:
        raw_data_df (DataFrame): raw data
        columns (list): selected metrics to plot the boxplot
    """
    if len(raw_data_df) == 0:
        logger.error('DataAnalyzer: empty data for boxplot.')
        return
    data_columns = raw_data_df.columns
    for column in columns:
        if column not in data_columns:
            logger.warning('DataAnalyzer: invalid column {} for boxplot.'.format(column))
            columns.remove(column)
    temp_raw_data_df = raw_data_df.fillna(value=raw_data_df.mean())
    n = len(columns)
    for i in range(n):
        sns.set(style='whitegrid')
        plt.subplot(n, 1, i + 1)
        sns.boxplot(x=columns[i], data=temp_raw_data_df, orient='h')
    plt.subplots_adjust(hspace=1)
    plt.show()
