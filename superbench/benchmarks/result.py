# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json


class BenchmarkResult():
    '''Result class of all benchmarks, defines the result format.

    Args:
        name: name of benchmark.
        return_code: return code of benchmark.
        run_count: run count of benchmark, all runs will be organized as array.
        start_time: starting time of benchmark.
        end_time: ending time of benchmark.
        raw_data: keys are metrics, values are arrays for N runs.
        result: keys are metrics, values are arrays for N runs.
    '''
    def __init__(self, name, run_count=0):
        self.name = name
        self.run_count = run_count
        self.return_code = 0
        self.start_time = None
        self.end_time = None
        self.raw_data = dict()
        self.result = dict()

    def __eq__(self, rhs):
        return self.__dict__ == rhs.__dict__

    def add_raw_data(self, metric, value):
        if metric not in self.raw_data:
            self.raw_data[metric] = list()
        self.raw_data[metric].append(value)

    def add_result(self, metric, value):
        if metric not in self.result:
            self.result[metric] = list()
        self.result[metric].append(value)

    def set_timestamp(self, start, end):
        self.start_time = start
        self.end_time = end

    def to_string(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_string(cls, string):
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
