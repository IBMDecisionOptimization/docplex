# --------------------------------------------------------------------------
# File: _multiobj.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Multi-Objective API"""
from ._baseinterface import BaseInterface
from ._subinterfaces import ObjSense
from . import _procedural as _proc
from . import _aux_functions as _aux
from . import _matrices as _mat
from . import _pycplex as CPX

class _Pair():
    def __init__(self):
        self.first = None
        self.second = None


class MultiObjInterface(BaseInterface):
    """Methods for adding, querying, and modifying multiple objectives.

    The methods in this interface can be used to add, query, and modify
    objectives in a specified problem. These objectives are used when
    multi-objective optimization is initiated.

    See also `MultiObjSolnInterface` where methods for accessing
    solutions for multi-objective models can be found.

    For more details see the section on multi-objective optimization in
    the CPLEX User's Manual.
    """

    sense = ObjSense()
    """See `ObjSense()`"""

    def __init__(self, cpx):
        """Creates a new MultiObjInterface.

        The Multi-Objective interface is exposed by the top-level `Cplex`
        class as `Cplex.multiobj`. This constructor is not meant to be
        used externally.
        """
        super().__init__(cplex=cpx, getindexfunc=_proc.multiobjgetindex)

    def get_num(self):
        """Returns the number of objectives in the problem.

        See :cpxapi:`CPXgetnumobjs` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.multiobj.get_num()
        1
        >>> indices = c.multiobj.set_num(2)
        >>> c.multiobj.get_num()
        2
        """
        return _proc.getnumobjs(self._env._e, self._cplex._lp)

    def set_num(self, numobj):
        """Sets the number of objectives in the problem instance.

        There is always at least one objective in the problem instance
        (indexed 0) thus numobj must be at least 1. If before calling
        this method there were more objectives in the instance than the
        specified numobj then the objectives whose index is >= numobj are
        removed from the instance. If before calling this method the
        number of objectives was <= numobj then new objectives are
        created, all with all-zero coefficients and default settings
        (like priority, weight, etc).

        See :cpxapi:`CPXsetnumobjs` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.multiobj.set_num(2)
        >>> c.multiobj.get_num()
        2
        """
        _proc.setnumobjs(self._env._e, self._cplex._lp, numobj)

    def get_names(self, *args):
        """Returns the names of a set of objectives.

        There are four forms by which multiobj.get_names may be called.

        multiobj.get_names()
          return the names of all objectives from the problem.

        multiobj.get_names(i)
          i must be an objective index. Returns the name of row i.

        multiobj.get_names(s)
          s must be a sequence of objective indices. Returns the names of
          the objectives with indices the members of s. Equivalent to
          [multiobj.get_names(i) for i in s]

        multiobj.get_names(begin, end)
          begin and end must be objective indices. Returns the names of
          the objectives with indices between begin and end, inclusive of
          end. Equivalent to multiobj.get_names(range(begin, end + 1)).

        See :cpxapi:`CPXmultiobjgetname` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.multiobj.set_definition(0, name='mo1')
        >>> c.multiobj.get_names(0)
        'mo1'
        """
        def _get_name(objidx):
            return _proc.multiobjgetname(self._env._e, self._cplex._lp,
                                         objidx)
        return _aux.apply_freeform_one_arg(_get_name, self._conv,
                                           self.get_num(), args)

    def set_name(self, objidx, name):
        """Sets the name of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.multiobj.set_num(3)
        >>> for i in range(3):
        ...     c.multiobj.set_name(i, str(i))
        >>> c.multiobj.get_names()
        ['0', '1', '2']
        """
        objidx = self._conv(objidx)
        _proc.multiobjchgattribs(self._env._e, self._cplex._lp,
                                 objidx, name=name)

    def get_definition(self, objidx, begin=None, end=None):
        """Returns the definition of an objective.

        Returns an objective definitions, where the definition is a list
        containing the following components: obj (a list containing the
        linear objective coefficients), offset, weight, priority, abstol,
        reltol (see `set_definition`).

        objidx is the name or index of the objective to be accessed.

        The optional begin and end arguments must be variable indices
        or names. Together, begin and end specify the range of objective
        function coefficients to be returned. By default, the linear
        objective coefficients of all variables from the problem will be
        returned (i.e., begin will default to the first variable index
        and end will default to the last variable index).

        See :cpxapi:`CPXmultiobjgetobj` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> varind = list(c.variables.add(obj=[1.0, 2.0]))
        >>> c.multiobj.get_definition(0)
        [[1.0, 2.0], 0.0, 1.0, 0, 0.0, 0.0]
        """
        objidx = self._conv(objidx)
        varconv = self._cplex.variables._conv
        if begin is None:
            begin = 0
        else:
            begin = varconv(begin)
        if end is None:
            end = self._cplex.variables.get_num() - 1
        else:
            end = varconv(end)
        return _proc.multiobjgetobj(self._env._e, self._cplex._lp,
                                    objidx, begin, end)

    def set_definition(self, objidx, obj=None, offset=0.0, weight=1.0,
                       priority=0, abstol=None, reltol=None, name=None):
        """Sets the definition of an objective.

        multiobj.set_definition accepts the keyword arguments objidx,
        obj, offset, weight, priority, abstol, reltol, and name.

        objidx is the name or index of the objective to be set. The
        objective index must be in the interval
        [0, Cplex.multiobj.get_num() - 1].

        obj can be either a SparsePair or a list of two lists specifying
        the linear component of the objective. If not specified, the
        coefficients of every variable are set to 0.0.

        Note
          obj must not contain duplicate indices. If obj references a
          variable more than once, either by index, name, or a
          combination of index and name, an exception will be raised.

        offset is the offset of the objective to be set. If not
        specififed, the offset is set to 0.0.

        weight is the weight of the objective to be set. For the
        definition of the weight see the description of blended objective
        in the multi-objective optimization section of the CPLEX User's
        Manual. If not specified, the weight is set to 1.0.

        priority is the priority of the objective to be set. It must be a
        nonnegative integer. For the definition of the priority see the
        description of lexicographic objective in the multi-objective
        optimization section of the CPLEX User's Manual. If not
        specified, the priority is set to 0.

        abstol is the absolute tolerance of the objective to be set. If
        not specified, the absolute tolerance is set to 0.0.

        reltol is the relative tolerance of the objective to be set. If
        not specified, the relative tolerance is set to 0.0.

        name is a string representing the name of the objective to be
        set. If not specified, the objective name will default to None.

        See :cpxapi:`CPXmultiobjsetobj` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> varind = list(c.variables.add(names=['x1', 'x2']))
        >>> c.multiobj.set_definition(
        ...     objidx=0,
        ...     obj=cplex.SparsePair(ind=varind, val=[1.0, 2.0]),
        ...     offset=0.0,
        ...     weight=1.0,
        ...     priority=0,
        ...     abstol=1e-06,
        ...     reltol=1e-04,
        ...     name='obj1')
        >>> c.multiobj.get_definition('obj1')
        [[1.0, 2.0], 0.0, 1.0, 0, 1e-06, 0.0001]
        >>> c.multiobj.get_names(0)
        'obj1'
        """
        if obj is None:
            obj = _mat.SparsePair()
        objind, objval = _mat.unpack_pair(obj)
        objind = self._cplex.variables._conv(objind)
        if abstol is None:
            abstol = self._cplex.parameters.mip.tolerances.absmipgap.default()
        if reltol is None:
            reltol = self._cplex.parameters.mip.tolerances.mipgap.default()
        _proc.multiobjsetobj(self._env._e, self._cplex._lp, objidx, objind,
                             objval, offset, weight, priority, abstol, reltol,
                             name)

    def get_linear(self, objidx, *args):
        """Returns the linear coefficients of a set of variables.

        Can be called by four forms each of which requires an objidx
        argument. objidx must be an objective name or index.

        multiobj.get_linear(objidx)
          return the linear objective coefficients of all variables
          from the problem.

        multiobj.get_linear(objidx, i)
          i must be a variable name or index. Returns the linear
          objective coefficient of the variable whose index or name is i.

        multiobj.get_linear(objidx, s)
          s must be a sequence of variable names or indices. Returns the
          linear objective coefficient of the variables with indices the
          members of s. Equivalent to
          [multiobj.get_linear(objidx, i) for i in s]

        multiobj.get_linear(objidx, begin, end)
          begin and end must be variable indices or variable names.
          Returns the linear objective coefficient of the variables with
          indices between begin and end, inclusive of end. Equivalent to
          multiobj.get_linear(objidx, range(begin, end + 1)).

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(
        ...     obj=[1.5 * i for i in range(10)],
        ...     names=[str(i) for i in range(10)])
        >>> c.variables.get_num()
        10
        >>> c.multiobj.get_linear(0, 8)
        12.0
        >>> c.multiobj.get_linear(0, '1', 3)
        [1.5, 3.0, 4.5]
        >>> c.multiobj.get_linear(0, [2, '0', 5])
        [3.0, 0.0, 7.5]
        >>> c.multiobj.get_linear(0)
        [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
        """
        (coeffs, _, _, _, _, _) = self.get_definition(objidx)
        def getcoeffs(begin, end=self._cplex.variables.get_num() - 1):
            return CPX._getArrayView(coeffs, begin, end + 1)
        return _aux.apply_freeform_two_args(
            getcoeffs, self._cplex.variables._conv, args)

    def set_linear(self, objidx, *args):
        """Changes the linear part of an objective function.

        Can be called by two forms each of which requires an objidx
        argument. objidx must be an objective name or index.

        multiobj.set_linear(objidx, var, value)
          var must be a variable index or name and value must be a float.
          Changes the coefficient of the variable identified by var to
          value.

        multiobj.set_linear(objidx, sequence)
          sequence is a sequence of pairs (var, value) as described
          above. Changes the coefficients for the specified variables to
          the given values.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(names=[str(i) for i in range(4)])
        >>> c.multiobj.get_linear(0)
        [0.0, 0.0, 0.0, 0.0]
        >>> c.multiobj.set_linear(0, 0, 1.0)
        >>> c.multiobj.get_linear(0)
        [1.0, 0.0, 0.0, 0.0]
        >>> c.multiobj.set_linear(0, '3', -1.0)
        >>> c.multiobj.get_linear(0)
        [1.0, 0.0, 0.0, -1.0]
        >>> c.multiobj.set_linear(0, [('2', 2.0), (1, 0.5)])
        >>> c.multiobj.get_linear(0)
        [1.0, 0.5, 2.0, -1.0]
        """
        objidx = self._conv(objidx)
        pair = _Pair()
        # NB: pair.first and pair.second get set as a side effect of
        #     running apply_pairs below!
        def set_pair(first, second):
            pair.first = first
            pair.second = second
        _aux.apply_pairs(set_pair, self._cplex.variables._conv, *args)
        ncols = self._cplex.variables.get_num()
        allind = list(range(ncols))
        # To preserve the values that have not been provided, we query
        # current objective. This is to maintain consistent semantics
        # with Cplex.objective.set_linear.
        allval = self._cplex.multiobj.get_linear(objidx)
        for idx, val in zip(pair.first, pair.second):
            allval[idx] = val
        _proc.multiobjsetobj(self._env._e, self._cplex._lp, objidx,
                             objind=allind, objval=allval)

    def get_sense(self):
        """Returns the sense of all objective functions.

        See `ObjSense`.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.multiobj.sense[c.multiobj.get_sense()]
        'minimize'
        >>> c.multiobj.set_sense(c.multiobj.sense.maximize)
        >>> c.multiobj.sense[c.multiobj.get_sense()]
        'maximize'
        >>> c.multiobj.set_sense(c.multiobj.sense.minimize)
        >>> c.multiobj.sense[c.multiobj.get_sense()]
        'minimize'
        """
        return self._cplex.objective.get_sense()

    def set_sense(self, sense):
        """Sets the sense of all objective functions.

        Note
          All objective functions share the same sense. To model an
          objective with a different sense use a negative value for the
          weight attribute. See `set_weight`.

        The argument to this method must be either `ObjSense.minimize`
        or `ObjSense.maximize`.

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.multiobj.sense[c.multiobj.get_sense()]
        'minimize'
        >>> c.multiobj.set_sense(c.multiobj.sense.maximize)
        >>> c.multiobj.sense[c.multiobj.get_sense()]
        'maximize'
        >>> c.multiobj.set_sense(c.multiobj.sense.minimize)
        >>> c.multiobj.sense[c.multiobj.get_sense()]
        'minimize'
        """
        self._cplex.objective.set_sense(sense)

    def get_offset(self, objidx):
        """Returns the constant offset of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.get_offset(0)
        0.0
        """
        (_, offset, _, _, _, _) = self.get_definition(objidx)
        return offset

    def set_offset(self, objidx, offset):
        """Sets the constant offset of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.objective.set_offset(3.14)
        >>> c.objective.get_offset()
        3.14
        """
        objidx = self._conv(objidx)
        _proc.multiobjchgattribs(self._env._e, self._cplex._lp,
                                 objidx, offset=offset)

    def get_weight(self, objidx):
        """Returns the weight of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.get_weight(0)
        1.0
        """
        (_, _, weight, _, _, _) = self.get_definition(objidx)
        return weight

    def set_weight(self, objidx, weight):
        """Sets the weight of an objective function.

        objidx must be an objective name or index.

        Note
          All objective functions share the same sense. To model an
          objective with a different sense use a negative value for the
          weight attribute. See `set_sense`.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.set_weight(0, -2.0)
        >>> c.multiobj.get_weight(0)
        -2.0
        """
        objidx = self._conv(objidx)
        _proc.multiobjchgattribs(self._env._e, self._cplex._lp,
                                 objidx, weight=weight)

    def get_priority(self, objidx):
        """Returns the priority of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.get_priority(0)
        0
        """
        (_, _, _, priority, _, _) = self.get_definition(objidx)
        return priority

    def set_priority(self, objidx, priority):
        """Sets the priority of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.set_priority(0, 2)
        >>> c.multiobj.get_priority(0)
        2
        """
        objidx = self._conv(objidx)
        _proc.multiobjchgattribs(self._env._e, self._cplex._lp,
                                 objidx, priority=priority)

    def get_abstol(self, objidx):
        """Returns the absolute tolerance of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.get_abstol(0)
        0.0
        """
        (_, _, _, _, abstol, _) = self.get_definition(objidx)
        return abstol

    def set_abstol(self, objidx, abstol):
        """Sets the absolute tolerance of an objective function.

        objidx must be an objective name or index.

        abstol should be a float. When specifying a new value, the same
        limits apply as with the
        Cplex.parameters.mip.tolerances.absmipgap parameter. See the
        section on Specifying multiple objective problems in the CPLEX
        User's Manual for the details on the meaning of this tolerance.

        See :cpxapi:`CPXmultiobjchgattribs` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.set_abstol(0, 1e-6)
        >>> c.multiobj.get_abstol(0)
        1e-06
        """
        objidx = self._conv(objidx)
        _proc.multiobjchgattribs(self._env._e, self._cplex._lp,
                                 objidx, abstol=abstol)

    def get_reltol(self, objidx):
        """Returns the relative tolerance of an objective function.

        objidx must be an objective name or index.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.get_reltol(0)
        0.0
        """
        (_, _, _, _, _, reltol) = self.get_definition(objidx)
        return reltol

    def set_reltol(self, objidx, reltol):
        """Sets the relative tolerance of an objective function.

        objidx must be an objective name or index.

        reltol should be a float. When specifying a new value, the same
        limits apply as with the Cplex.parameters.mip.tolerances.mipgap
        parameter. Note that a nondefault setting of this parameter only
        applies to MIP multiobjective problems. See the section on
        Specifying multiple objective problems in the CPLEX User's Manual
        for the details on the meaning of this tolerance.

        See :cpxapi:`CPXmultiobjchgattribs` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0 for i in range(3)])
        >>> c.multiobj.set_reltol(0, 1e-4)
        >>> c.multiobj.get_reltol(0)
        0.0001
        """
        objidx = self._conv(objidx)
        _proc.multiobjchgattribs(self._env._e, self._cplex._lp,
                                 objidx, reltol=reltol)
