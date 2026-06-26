# --------------------------------------------------------------------------
# File: _baseinterface.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Base-interface of the CPLEX API"""
import weakref
from . import _aux_functions as _aux


class BaseInterface():
    """Common methods for sub-interfaces."""

    def __init__(self, cplex, advanced=False, getindexfunc=None):
        """Creates a new BaseInterface.

        This class is not meant to be instantiated directly nor used
        externally.
        """
        if advanced:
            self._cplex = cplex
        else:
            self._cplex = weakref.proxy(cplex)
        self._env = weakref.proxy(cplex._env)
        self._get_index_function = getindexfunc

    def _conv(self, name, cache=None):
        """Converts from names to indices as necessary."""
        return _aux.convert(name, self._get_index, cache)

    @staticmethod
    def _add_iter(getnumfun, addfun, *args, **kwargs):
        """non-public"""
        old = getnumfun()
        addfun(*args, **kwargs)
        return range(old, getnumfun())

    @staticmethod
    def _add_single(getnumfun, addfun, *args, **kwargs):
        """non-public"""
        addfun(*args, **kwargs)
        return getnumfun() - 1  # minus one for zero-based indices

    def _get_index(self, name):
        return self._get_index_function(
            self._env._e, self._cplex._lp, name)

    def get_indices(self, name):
        """Converts from names to indices.

        If name is a string, get_indices returns the index of the
        object with that name.  If no such object exists, an
        exception is raised.

        If name is a sequence of strings, get_indices returns a list
        of the indices corresponding to the strings in name.
        Equivalent to map(self.get_indices, name).

        If the subclass does not provide an index function (i.e., the
        interface is not indexed), then a NotImplementedError is raised.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(names=["a", "b"])
        >>> c.variables.get_indices("a")
        0
        >>> c.variables.get_indices(["a", "b"])
        [0, 1]
        """
        if self._get_index_function is None:
            raise NotImplementedError("This is not an indexed interface")
        if isinstance(name, str):
            return self._get_index(name)
        return [self._get_index(x) for x in name]
