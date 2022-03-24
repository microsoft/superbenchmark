# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for result summary ops."""

from typing import Dict, Callable
import numbers

from superbench.benchmarks.context import Enum
from superbench.common.utils import logger


class SummaryType(Enum):
    """The Enum class representing different summary ops."""

    MEAN = 'mean'
    PENCENTILE = 'percentile'
    MIN = 'min'
    MAX = 'max'
    STD = 'std'
    COUNT = 'count'


class SummaryOp:
    """SummaryOp class to maintain all summary functions."""

    functions: Dict[SummaryType, Callable] = dict()

    @classmethod
    def add_summary_func(cls, summary_type):
        """Add summary fuction.

        Args:
            summary_type (SummaryType): The type of summary function.

        Return:
            decorator (Callable): return the decorator to add the summary function.
        """
        def decorator(func):
            cls.functions[summary_type] = func
            return func

        return decorator

    @classmethod
    def get_summary_func(cls, summary_type):
        """Get summary fuction by summary_type.

        Args:
            summary_type (SummaryType): The type of summary function.

        Return:
            func (Callable): summary function, None means invalid summary type.
        """
        if summary_type in cls.functions:
            return cls.functions[summary_type]

        return None

    @staticmethod
    def _check_raw_data_df(raw_data_df):
        """Check whether raw_data_df is empty or None.

        Args:
            raw_data_df (DataFrame): raw data df
        """
        if raw_data_df is None or raw_data_df.empty:
            logger.log_and_raise(exception=Exception, msg='empty data in summary op')

    @staticmethod
    def mean(raw_data_df):
        """Mean of raw_data_df.

        Args:
            raw_data_df (DataFrame): raw data df

        Returns:
            Series: mean of raw_data_df
        """
        SummaryOp._check_raw_data_df(raw_data_df)
        return raw_data_df.mean()

    @staticmethod
    def percentile(raw_data_df, val):
        """Pencentile$(val) of raw_data_df.

        Args:
            raw_data_df (DataFrame): raw data df
            val (numbers.Number): the pencentile value, 1-99

        Returns:
            Series: mean of raw_data_df
        """
        SummaryOp._check_raw_data_df(raw_data_df)
        if not isinstance(val, numbers.Number) or val < 1 or val > 99:
            logger.log_and_raise(exception=Exception, msg='val in pencentile should be 1-99')
        return raw_data_df.quantile(val / 100)

    @staticmethod
    def min(raw_data_df):
        """The min of values for each column in raw_data_df.

        Args:
            raw_data_df (DataFrame): raw data df

        Returns:
            Series: min of raw_data_df
        """
        SummaryOp._check_raw_data_df(raw_data_df)
        return raw_data_df.min()

    @staticmethod
    def max(raw_data_df):
        """The max of values for each column in raw_data_df.

        Args:
            raw_data_df (DataFrame): raw data df

        Returns:
            Series: max of raw_data_df
        """
        SummaryOp._check_raw_data_df(raw_data_df)
        return raw_data_df.max()

    @staticmethod
    def std(raw_data_df):
        """The std of values for each column in raw_data_df.

        Args:
            raw_data_df (DataFrame): raw data df

        Returns:
            Series: std of raw_data_df
        """
        SummaryOp._check_raw_data_df(raw_data_df)
        return raw_data_df.std(axis=0, skipna=True)

    @staticmethod
    def count(raw_data_df):
        """The number of values for each column in raw_data_df.

        Args:
            raw_data_df (DataFrame): raw data df

        Returns:
            Series: count of raw_data_df
        """
        SummaryOp._check_raw_data_df(raw_data_df)
        return raw_data_df.count()


SummaryOp.add_summary_func(SummaryType.MEAN)(SummaryOp.mean)
SummaryOp.add_summary_func(SummaryType.PENCENTILE)(SummaryOp.percentile)
SummaryOp.add_summary_func(SummaryType.MIN)(SummaryOp.min)
SummaryOp.add_summary_func(SummaryType.MAX)(SummaryOp.max)
SummaryOp.add_summary_func(SummaryType.STD)(SummaryOp.std)
SummaryOp.add_summary_func(SummaryType.COUNT)(SummaryOp.count)
