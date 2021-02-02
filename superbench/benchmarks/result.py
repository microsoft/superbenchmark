# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for unified result of benchmarks."""

import json

from superbench.common.utils import logger


class BenchmarkResult():
    """Result class of all benchmarks.

    Defines the unified result format.
    """
    def __init__(self, name, benchmark_type, run_count=0):
        """Constructor.

        Args:
            name (str): name of benchmark.
            benchmark_type (str): type of the benchmark, such as model, micro or docker.
            run_count (int): run count of benchmark,
              all runs will be organized as array.
        """
        self.name = name
        self.type = benchmark_type
        self.run_count = run_count
        self.return_code = 0
        self.start_time = None
        self.end_time = None
        self.raw_data = dict()
        self.result = dict()

    def __eq__(self, rhs):
        """Override equal function for deep comparison.

        Args:
            rhs (BenchmarkResult): instance to compare.

        Return:
            True if two instances have all the same values
              for all the same attributes.
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
                'metric name of benchmark is not string, name: {}, metric type: {}'.format(self.name, type(metric))
            )
            return False

        if metric not in self.raw_data:
            self.raw_data[metric] = list()
        self.raw_data[metric].append(value)

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
                'metric name of benchmark is not string, name: {}, metric type: {}'.format(self.name, type(metric))
            )
            return False

        if metric not in self.result:
            self.result[metric] = list()
        self.result[metric].append(value)

        return True

    def set_timestamp(self, start, end):
        """Set the start and end timestamp of benchmarking.

        Args:
            start (datetime): start timestamp of benchmarking.
            end (datetime): end timestamp of benchmarking.
        """
        self.start_time = start
        self.end_time = end

    def to_string(self):
        """Serialize the BenchmarkResult object to string.

        Return:
            The serialized string of BenchmarkResult object.
        """
        return json.dumps(self.__dict__)

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
                if field not in obj:
                    return None
                else:
                    ret.__dict__[field] = obj[field]

        return ret
