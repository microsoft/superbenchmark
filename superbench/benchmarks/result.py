# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for unified result of benchmarks."""

import os
import json
from enum import Enum

from superbench.common.utils import logger


class BenchmarkResult():
    """Result class of all benchmarks.

    Defines the unified result format.
    """
    def __init__(self, name, type, return_code, run_count=0):
        """Constructor.

        Args:
            name (str): name of benchmark.
            type (BenchmarkType): type of benchmark.
            return_code (ReturnCode): return code of benchmark.
            run_count (int): run count of benchmark, all runs will be organized as array.
        """
        self.__name = name
        self.__type = type
        self.__run_count = run_count
        self.__return_code = return_code
        self.__start_time = None
        self.__end_time = None
        self.__raw_data = dict()
        self.__result = dict()
        self.__result['return_code'] = [return_code.value]
        self.__reduce_op = dict()
        self.__reduce_op['return_code'] = None

    def __eq__(self, rhs):
        """Override equal function for deep comparison.

        Args:
            rhs (BenchmarkResult): instance to compare.

        Return:
            True if two instances have all the same values for all the same attributes.
        """
        return self.__dict__ == rhs.__dict__

    def add_raw_data(self, metric, value, log_raw_data):
        """Add raw benchmark data into result.

        Args:
            metric (str): metric name which is the key.
            value (str or list): raw benchmark data.
              For e2e model benchmarks, its type is list.
              For micro-benchmarks or docker-benchmarks, its type is string.
            log_raw_data (bool): whether to log raw data into file instead of saving it into result object.

        Return:
            True if succeed to add the raw data.
        """
        if not metric or not isinstance(metric, str):
            logger.error(
                'metric name of benchmark is not string, name: {}, metric type: {}'.format(self.__name, type(metric))
            )
            return False

        if log_raw_data:
            with open(os.path.join(os.getcwd(), 'rawdata.log'), 'a') as f:
                f.write('metric:{}\n'.format(metric))
                f.write('rawdata:{}\n\n'.format(value))
        else:
            if metric not in self.__raw_data:
                self.__raw_data[metric] = list()
            self.__raw_data[metric].append(value)

        return True

    def add_result(self, metric, value, reduce_type=None):
        """Add summarized data into result.

        Args:
            metric (str): metric name which is the key.
            value (float): summarized data.
              For e2e model benchmarks, the value is step-time or throughput.
              For micro-benchmarks, the value is FLOPS, bandwidth and etc.
            reduce_type (ReduceType): type of reduce function.

        Return:
            True if succeed to add the result.
        """
        if not metric or not isinstance(metric, str):
            logger.error(
                'metric name of benchmark is not string, name: {}, metric type: {}'.format(self.__name, type(metric))
            )
            return False

        if metric not in self.__result:
            self.__result[metric] = list()
            self.__reduce_op[metric] = reduce_type.value if isinstance(reduce_type, Enum) else None
        self.__result[metric].append(value)

        return True

    def set_timestamp(self, start, end):
        """Set the start and end timestamp of benchmarking.

        Args:
            start (datetime): start timestamp of benchmarking.
            end (datetime): end timestamp of benchmarking.
        """
        self.__start_time = start
        self.__end_time = end

    def set_benchmark_type(self, benchmark_type):
        """Set the type of benchmark.

        Args:
            benchmark_type (BenchmarkType): type of benchmark, such as BenchmarkType.MODEL, BenchmarkType.MICRO.
        """
        self.__type = benchmark_type

    def set_return_code(self, return_code):
        """Set the return code.

        Args:
            return_code (ReturnCode): return code defined in superbench.benchmarks.ReturnCode.
        """
        self.__return_code = return_code
        self.__result['return_code'][0] = return_code.value

    def to_string(self):
        """Serialize the BenchmarkResult object to string.

        Return:
            The serialized string of BenchmarkResult object.
        """
        formatted_obj = dict()
        for key in self.__dict__:
            # The name of internal member is like '_BenchmarkResult__name'.
            # For the result object return to caller, just keep 'name'.
            formatted_key = key.split('__')[1]
            if isinstance(self.__dict__[key], Enum):
                formatted_obj[formatted_key] = self.__dict__[key].value
            else:
                formatted_obj[formatted_key] = self.__dict__[key]

        return json.dumps(formatted_obj)

    @property
    def name(self):
        """Decoration function to access __name."""
        return self.__name

    @property
    def type(self):
        """Decoration function to access __type."""
        return self.__type

    @property
    def run_count(self):
        """Decoration function to access __run_count."""
        return self.__run_count

    @property
    def return_code(self):
        """Decoration function to access __return_code."""
        return self.__return_code

    @property
    def default_metric_count(self):
        """Decoration function to get the count of default metrics."""
        count = 0
        if 'return_code' in self.__result:
            count += 1

        return count

    @property
    def start_time(self):
        """Decoration function to access __start_time."""
        return self.__start_time

    @property
    def end_time(self):
        """Decoration function to access __end_time."""
        return self.__end_time

    @property
    def raw_data(self):
        """Decoration function to access __raw_data."""
        return self.__raw_data

    @property
    def result(self):
        """Decoration function to access __result."""
        return self.__result

    @property
    def reduce_op(self):
        """Decoration function to access __reduce_op."""
        return self.__reduce_op
