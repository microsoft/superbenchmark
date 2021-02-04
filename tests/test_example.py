# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test example.

Get it from https://docs.pytest.org/en/stable/.
"""

import superbench


def inc(x):
    """Increase an integer.

    Args:
        x (int): Input value.

    Returns:
        int: Increased value.
    """
    return x + 1


def test_answer():
    """Test inc function."""
    assert inc(3) == 4


def test_superbench():
    """Test SuperBench."""
    assert (superbench.__version__ == '0.0.0')
    assert (superbench.__author__ == 'Microsoft')
