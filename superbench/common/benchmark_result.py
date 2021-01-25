# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json


class Result():
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
    def __init__(self, name='', return_code=0, run_count=0, start_time=0,
                 end_time=0, raw_data=dict(), result=dict()):
        self.name = name
        self.return_code = return_code
        self.run_count = run_count
        self.start_time = start_time
        self.end_time = end_time
        self.raw_data = raw_data
        self.result = result

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
