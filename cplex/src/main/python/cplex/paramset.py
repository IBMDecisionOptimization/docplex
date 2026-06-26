# --------------------------------------------------------------------------
# File: paramset.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""ParameterSet API"""
import weakref
from ._internal import _procedural as _proc


def _get_id(param):
    """Returns a parameter ID.

    If param is a Parameter object, then we get the ID from it.
    Otherwise, we assume param is an integer.
    """
    try:
        # If this is a Parameter object, then return its _id attr.
        return param._id
    except AttributeError:
        # Otherwise, we assume it's an integer.
        return param


def _get_type(env, param):
    """Returns a parameter type.

    If param is a Parameter object, then we get the type from it.
    Otherwise, we assume param is an integer and query the parameter
    type.
    """
    try:
        return param._type
    except AttributeError:
        return _proc.getparamtype(env, param)


class ParameterSet():
    """A parameter set object for use with multi-objective optimization.

    A parameter set consists of key-value pairs where the key is a CPLEX
    parameter ID (e.g., CPX_PARAM_ADVIND) and the value is the associated
    parameter value.

    When adding, getting, or deleting items from a parameter set the
    param argument can be either a Parameter object (e.g,
    Cplex.parameters.advance) or an integer ID (e.g., CPX_PARAM_ADVIND
    (1001)).

    For more details see the section on multi-objective optimization in
    the CPLEX User's Manual.

    See `Cplex.create_parameter_set` and `Cplex.copy_parameter_set`.

    Example usage:

    >>> import cplex
    >>> c = cplex.Cplex()
    >>> ps = c.create_parameter_set()
    >>> ps.add(c.parameters.advance, c.parameters.advance.values.none)
    >>> len(ps)
    1
    """

    def __init__(self, env):
        """Constructor of the ParameterSet class.

        This class is not meant to be instantiated directly nor used
        externally.
        """
        self._disposed = False
        self._env = weakref.proxy(env)
        self._ps = _proc.paramsetcreate(self._env._e)

    def _throw_if_disposed(self):
        if self._disposed:
            raise ValueError(
                'illegal method invocation after ParameterSet.end()')

    def end(self):
        """Releases the ParameterSet object.

        Frees all data structures associated with a ParameterSet. After
        a call of the method end(), the invoking object can no longer be
        used. Attempts to use them subsequently raise a ValueError.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.end()
        """
        if self._disposed:
            return
        self._disposed = True
        try:
            _proc.paramsetfree(self._env._e, self._ps)
        except ReferenceError:
            # Ignore error raised if the reference env of our weakref
            # has been garbage collected. If the env has already been
            # closed, then the paramset has already been freed.
            pass
        self._ps = None

    def __del__(self):
        """Destructor of the ParameterSet class.

        When a ParameterSet object is destoyed, the end() method is
        called.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> del ps
        """
        self.end()

    def __enter__(self):
        """Enter the runtime context related to this object.

        The with statement will bind this method's return value to the
        target specified in the as clause of the statement, if any.

        ParameterSet objects return themselves.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> with c.create_parameter_set():
        ...     pass  # do something here
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context.

        When we exit the with block, the end() method is called.
        """
        self.end()

    def add(self, param, value):
        """Add a parameter ID and value to a parameter set.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        """
        self._throw_if_disposed()
        whichparam = _get_id(param)
        paramtype = _get_type(self._env._e, param)
        _proc.paramsetadd(self._env._e, self._ps, whichparam, value,
                          paramtype)

    def get(self, param):
        """Gets a parameter value.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        >>> val = ps.get(c.parameters.advance)
        >>> val == c.parameters.advance.values.none
        True
        """
        self._throw_if_disposed()
        whichparam = _get_id(param)
        paramtype = _get_type(self._env._e, param)
        return _proc.paramsetget(self._env._e, self._ps, whichparam,
                                 paramtype)

    def get_ids(self):
        """Gets the parameter IDs contained in a parameter set.

        Returns an iterator containing the parameter IDs in a parameter
        set.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        >>> list(ps.get_ids())
        [1001]
        """
        self._throw_if_disposed()
        return _proc.paramsetgetids(self._env._e, self._ps)

    def delete(self, param):
        """Deletes a parameter from a parameter set.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        >>> len(ps)
        1
        >>> ps.delete(c.parameters.advance)
        >>> len(ps)
        0
        """
        self._throw_if_disposed()
        _proc.paramsetdel(self._env._e, self._ps, _get_id(param))

    def clear(self):
        """Clears all items from the parameter set.

        Example Usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        >>> ps.clear()
        >>> len(ps)
        0
        """
        self._throw_if_disposed()
        for item in self.get_ids():
            self.delete(item)

    def __len__(self):
        """Return the number of items in the parameter set.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> len(ps)
        0
        """
        self._throw_if_disposed()
        return _proc.paramsetgetnum(self._env._e, self._ps)

    def read(self, filename):
        """Reads parameter names and settings from the file specified by
        filename and copies them into the parameter set.

        Note that the content of the parameter set is not cleared out
        before the parameters in the file are copied into the parameter
        set. The parameters are read from the file one by one and are
        added to the parameter set, or, if the parameter was already
        present in the set, then its value is updated.

        This routine reads and copies files in the PRM format, as created
        by Cplex.parameters.write. The PRM format is documented in the
        CPLEX File Formats Reference Manual.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.parameters.advance.set(c.parameters.advance.values.none)
        >>> c.parameters.write_file('example.prm')
        >>> ps = c.create_parameter_set()
        >>> ps.read('example.prm')
        >>> value = ps.get(c.parameters.advance)
        >>> value == c.parameters.advance.values.none
        True
        """
        self._throw_if_disposed()
        _proc.paramsetreadcopy(self._env._e, self._ps, filename)

    def write(self, filename):
        """Writes a parameter file that contains the parameters in the
        parameter set.

        This routine writes a file in a format suitable for reading by
        ParameterSet.read or by Cplex.parameters.read.

        The file is written in the PRM format which is documented in the
        CPLEX File Formats Reference Manual.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        >>> ps.write('example.prm')
        >>> c.parameters.read_file('example.prm')
        >>> value = c.parameters.advance.get()
        >>> value == c.parameters.advance.values.none
        True
        """
        self._throw_if_disposed()
        _proc.paramsetwrite(self._env._e, self._ps, filename)
