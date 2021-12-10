# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for data analysis module."""

from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd

import superbench.analyzer.data_analysis as data_analysis


class TestDataAnalysis(unittest.TestCase):
    """Test for DataAnalysis class."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        self.output_dir = str(Path(__file__).parent.resolve())
        self.fig = self.output_dir + '/boxplot.png'
        self.baseline = self.output_dir + '/baseline.json'

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        p = Path(self.fig)
        if p.is_file():
            p.unlink()
        p = Path(self.baseline)
        if p.is_file():
            p.unlink()

    def test_data_analysis(self):
        """Test for data analysis."""
        # Test - statistic
        co1 = np.random.rand(100)
        co2 = np.random.rand(100)
        co3 = np.random.rand(100)
        raw_data_df = pd.DataFrame({'a': co1, 'b': co2, 'c': co3})
        data_statistics_df = data_analysis.statistic(raw_data_df)
        assert (len(data_statistics_df) == 12)
        assert (len(data_statistics_df.columns) == 3)
        raw_data_df['d'] = ['a' for i in range(100)]
        data_statistics_df = data_analysis.statistic(raw_data_df)
        assert (len(data_statistics_df.columns) == 3)
        # Test - inter_quartile_range
        data_statistics_df = data_analysis.interquartile_range(raw_data_df)
        assert (len(data_statistics_df) == 20)
        assert (len(data_statistics_df.columns) == 3)
        # Test - correlation
        data_corr_df = data_analysis.correlation(raw_data_df)
        assert (len(data_corr_df) == 3)
        # Test - creat_boxplot
        data_analysis.creat_boxplot(raw_data_df, list(raw_data_df.columns), self.output_dir)
        fig = Path(self.fig)
        assert (fig.is_file())
        fig.unlink()
        # Test - generate baseline
        data_analysis.generate_baseline(raw_data_df, self.output_dir)
        baseline_path = Path(self.baseline)
        with baseline_path.open() as load_f:
            baseline = json.load(load_f)
        baseline_path.unlink()
        assert (len(baseline) == 3)
        # Test for invalid input
        raw_data_dict = {}
        assert (len(data_analysis.statistic(raw_data_dict)) == 0)
        assert (len(data_analysis.interquartile_range(raw_data_dict)) == 0)
        assert (len(data_analysis.correlation(raw_data_dict)) == 0)
