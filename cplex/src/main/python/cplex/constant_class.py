# --------------------------------------------------------------------------
# File: constant_class.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""A base class for classes that contain groups of constants."""


class ConstantClass():
    """A base class for classes that contain groups of constants."""

    def __init__(self):
        """Creates a new ConstantClass.

        This constructor is not meant to be used externally.
        """
        self.__constant_map = None

    def _get_constant_map(self):
        return {key: value
                for key, value
                in self.__class__.__dict__.items()
                if not key.startswith("_")}

    def __getitem__(self, item):
        """Converts a constant to a string."""
        if self.__constant_map is None:
            self.__constant_map = self._get_constant_map()
        for name, value in self.__constant_map.items():
            if item == value:
                return name
        raise KeyError(item)

    def __iter__(self):
        """Iterate over the constants in this class."""
        if self.__constant_map is None:
            self.__constant_map = self._get_constant_map()
        for value in self.__constant_map.values():
            yield value
