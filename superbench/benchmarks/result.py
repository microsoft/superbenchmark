# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for unified result of benchmarks."""

import json

from superbench.common.utils import logger


class BenchmarkResult():
    """Result class of all benchmarks.

    Defines the unified result format.
    """
    def __init__(self, name, type, run_count=0):
        """Constructor.

        Args:
            name (str): name of benchmark.
            type (str): type of benchmark.
            run_count (int): run count of benchmark, all runs will be organized as array.
        """
        self.__name = name
        self.__type = type
        self.__run_count = run_count
        self.__return_code = 0
        self.__start_time = None
        self.__end_time = None
        self.__raw_data = dict()
        self.__result = dict()

    def __eq__(self, rhs):
        """Override equal function for deep comparison.

        Args:
            rhs (BenchmarkResult): instance to compare.

        Return:
            True if two instances have all the same values for all the same attributes.
        """
        return self.__dict__ == rhs.__dict__

    def add_raw_data(self, metric, value):
        """Add raw benchmark data into result.

        Args:
            metric (str): metric name which is the key.
            value (str or list): raw benchmark data.
              For e2e model benchmarks, its type is list.
              For micro-benchmarks or docker-benchmarks, its type is string.

        Return:
            True if succeed to add the raw data.
        """
        if not metric or not isinstance(metric, str):
            logger.error(
                'metric name of benchmark is not string, name: {}, metric type: {}'.format(self.__name, type(metric))
            )
            return False

        if metric not in self.__raw_data:
            self.__raw_data[metric] = list()
        self.__raw_data[metric].append(value)

        return True

    def add_result(self, metric, value):
        """Add summarized data into result.

        Args:
            metric (str): metric name which is the key.
            value (float): summarized data.
              For e2e model benchmarks, the value is step-time or throughput.
              For micro-benchmarks, the value is FLOPS, bandwidth and etc.

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
            benchmark_type (str): type of benchmark, such as 'model', 'micro' or 'docker'.
        """
        self.__type = benchmark_type

    def to_string(self):
        """Serialize the BenchmarkResult object to string.

        Return:
            The serialized string of BenchmarkResult object.
        """
        formatted_obj = dict()
        for key in self.__dict__:
            # The name of internal member is like '_BenchmarkResult__name'.
            # For the result object return to caller, just keep 'name'.
            formatted_obj[key.split('__')[1]] = self.__dict__[key]

        return json.dumps(formatted_obj)

    @classmethod
    def from_string(cls, string):
        """Deserialize the string to BenchmarkResult object.

        Args:
            string (str): serialized string of BenchmarkResult object.

        Return:
            The deserialized BenchmarkResult object.
        """
        obj = json.loads(string)
        ret = None
        if 'name' in obj and 'run_count' in obj:
            ret = cls(obj['name'], obj['run_count'])
            fields = ret.__dict__.keys()
            for field in fields:
                # The name of internal member is like '_BenchmarkResult__name'.
                # For the result object return to caller, just keep 'name'.
                if field.split('__')[1] not in obj:
                    return None
                else:
                    ret.__dict__[field] = obj[field.split('__')[1]]

        return ret

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
