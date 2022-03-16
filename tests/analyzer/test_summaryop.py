# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for SummaryOp module."""

import unittest
from numpy import NaN, float64

import pandas as pd

from superbench.analyzer import SummaryOp, SummaryType


class TestSummaryOp(unittest.TestCase):
    """Test for Summary Ops."""
    def test_rule_op(self):
        """Test for defined rule operators."""
        # Test - get_rule_func
        # Negative case
        assert (not SummaryOp.get_summary_func('fake'))
        # Positive case
        summary_op = SummaryOp.get_summary_func(SummaryType.MEAN)
        assert (summary_op == SummaryOp.mean)
        summary_op = SummaryOp.get_summary_func(SummaryType.PENCENTILE)
        assert (summary_op == SummaryOp.percentile)
        summary_op = SummaryOp.get_summary_func(SummaryType.MIN)
        assert (summary_op == SummaryOp.min)
        summary_op = SummaryOp.get_summary_func(SummaryType.MAX)
        assert (summary_op == SummaryOp.max)
        summary_op = SummaryOp.get_summary_func(SummaryType.STD)
        assert (summary_op == SummaryOp.std)
        summary_op = SummaryOp.get_summary_func(SummaryType.COUNT)
        assert (summary_op == SummaryOp.count)

        # Test - _check_raw_data_Df
        # Negative case
        empty_data_df = pd.DataFrame()
        self.assertRaises(Exception, SummaryOp._check_raw_data_df, empty_data_df)
        self.assertRaises(Exception, SummaryOp._check_raw_data_df, None)

        data1 = [[1, 2, 3, 4], [4, 5, 6], [7, 8]]
        raw_data_df = pd.DataFrame(data1, columns=['a', 'b', 'c', 'd'])
        # Test - mean
        result = SummaryOp.mean(raw_data_df)
        expectedResult = pd.Series([4.0, 5.0, 4.5, 4.0], index=['a', 'b', 'c', 'd'])
        pd.testing.assert_series_equal(result, expectedResult)
        # Test - min
        result = SummaryOp.min(raw_data_df)
        expectedResult = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'], dtype=float64)
        pd.testing.assert_series_equal(result, expectedResult)
        # Test - max
        result = SummaryOp.max(raw_data_df)
        expectedResult = pd.Series([7, 8, 6, 4], index=['a', 'b', 'c', 'd'], dtype=float64)
        pd.testing.assert_series_equal(result, expectedResult)
        # Test - std
        result = SummaryOp.std(raw_data_df)
        print(result)
        expectedResult = pd.Series([3.0, 3.0, 2.1213203435596424, NaN], index=['a', 'b', 'c', 'd'], dtype=float64)
        pd.testing.assert_series_equal(result, expectedResult)
        # Test - count
        result = SummaryOp.count(raw_data_df)
        print(result)
        expectedResult = pd.Series([3, 3, 2, 1], index=['a', 'b', 'c', 'd'])
        pd.testing.assert_series_equal(result, expectedResult)
        # Test - pencentile
        result = SummaryOp.percentile(raw_data_df, 50)
        print(result)
        expectedResult = pd.Series([4.0, 5.0, 4.5, 4.0], index=['a', 'b', 'c', 'd'], dtype=float64)
        pd.testing.assert_series_equal(result, expectedResult, check_names=False)
        self.assertRaises(Exception, SummaryOp.percentile, 200)
