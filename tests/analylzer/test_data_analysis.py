# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for data analysis module."""

import numpy as np
import pandas as pd

import superbench.analyzer.data_analysis as data_analysis


def test_data_analysis():
    """Test for data analysis."""
    # Test - statistic
    co1 = np.random.rand(100)
    co2 = np.random.rand(100)
    co3 = np.random.rand(100)
    raw_data_df = pd.DataFrame({'a': co1, 'b': co2, 'c': co3})
    data_statistics_df = data_analysis.statistic(raw_data_df)
    assert (len(data_statistics_df) == 8)
    assert (len(data_statistics_df.columns) == 3)
    raw_data_df['d'] = ['a' for i in range(100)]
    data_statistics_df = data_analysis.statistic(raw_data_df)
    assert (len(data_statistics_df.columns) == 3)
    # Test - inter_quartile_range
    data_statistics_df = data_analysis.inter_quartile_range(raw_data_df)
    assert (len(data_statistics_df) == 16)
    assert (len(data_statistics_df.columns) == 3)
    # Test - correlation
    data_corr_df = data_analysis.correlation(raw_data_df)
    assert (len(data_corr_df) == 3)
