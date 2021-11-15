# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A module for data of monitor."""


class C:
    """Test."""
    def __init__(self):
        """Test."""
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        """Test."""
        self._x = value

    @x.deleter
    def x(self):
        """Test."""
        del self._x
