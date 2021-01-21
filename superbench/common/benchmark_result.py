# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json


class Result():
    def __init__(self, name='', errcode=0, run_count=0, start_time=0, end_time=0, raw_data=dict(), result=dict()):
        self.name = name
        self.errcode = errcode
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
        self.run_count += 1

    def add_result(self, metric, value):
        if metric not in self.result:
            self.result[metric] = list()
        self.result[metric].append(value)

    def set_timestamp(self, start, end):
        self.start_time = start
        self.end_time = end

    def to_string(self):
        return json.dumps(self.__dict__)
