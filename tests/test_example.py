# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# content of test_sample.py
# get it from https://docs.pytest.org/en/stable/


def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 4
