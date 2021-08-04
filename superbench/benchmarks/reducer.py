# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for result reducer."""

from typing import Dict, Callable
from statistics import mean

from superbench.benchmarks.context import Enum

class ReduceType(Enum):
    """The Enum class representing different reducer."""
    AVG = 'avg'
    MAX = 'max'
    MIN = 'min'
    SUM = 'sum'


class Reducer:
    """Reducer class to maintain all reduce functions."""
    functions: Dict[ReduceType, Callable] = dict()

    @classmethod
    def add_reduce_func(cls, reduce_type):
        """Add reduce fuction.

        Args:
            reduce_type (ReduceType): The type of reduce function.

        Return:
            decorator (Callable): return the decorator to add the reduce function.
        """
        def decorator(func):
            cls.functions[reduce_type] = func
            return func
        return decorator

    @classmethod
    def get_reduce_func(cls, reduce_type):
        """Get reduce fuction by reduce_type.

        Args:
            reduce_type (ReduceType): The type of reduce function.

        Return:
            func (Callable): reduce function, None means invalid reduce type.
        """
        if reduce_type in cls.functions:
            return cls.functions[reduce_type]

        return None


Reducer.add_reduce_func(ReduceType.MAX)(max)
Reducer.add_reduce_func(ReduceType.MIN)(min)
Reducer.add_reduce_func(ReduceType.SUM)(sum)
Reducer.add_reduce_func(ReduceType.AVG)(mean)
