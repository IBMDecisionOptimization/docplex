# --------------------------------------------------------------------------
# File: _multiobjsoln.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Multi-Objective Solution API"""
from . import _constants as _const
from . import _procedural as _proc
from ._baseinterface import BaseInterface
from ..constant_class import ConstantClass


class MultiObjFloatInfo(ConstantClass):
    """Types of floating point information that can be queried for
    multi-objective optimization.

    This class contains the types of solution information of type float
    that can be retreived from the solution of a sub-problem solved
    during multi-objective optimization.

    This information can be querried for each priority level with method
    `MultiObjSolnInterface.get_info`.
    """

    time = _const.CPX_MULTIOBJ_TIME
    """See :macros:`CPX_MULTIOBJ_TIME` in the C API."""

    dettime = _const.CPX_MULTIOBJ_DETTIME
    """See :macros:`CPX_MULTIOBJ_DETTIME` in the C API."""

    objective = _const.CPX_MULTIOBJ_OBJVAL
    """See :macros:`CPX_MULTIOBJ_OBJVAL` in the C API."""

    best_objective = _const.CPX_MULTIOBJ_BESTOBJVAL
    """See :macros:`CPX_MULTIOBJ_BESTOBJVAL` in the C API."""


class MultiObjIntInfo(ConstantClass):
    """Types of integer information that can be queried for
    multi-objective optimization.

    This class contains the types of solution information of type float
    that can be retreived from the solution of a sub-problem solved
    during multi-objective optimization.

    This information can be querried for each priority level with method
    `MultiObjSolnInterface.get_info`.
    """

    error = _const.CPX_MULTIOBJ_ERROR
    """See :macros:`CPX_MULTIOBJ_ERROR` in the C API."""

    status = _const.CPX_MULTIOBJ_STATUS
    """See :macros:`CPX_MULTIOBJ_STATUS` in the C API."""

    method = _const.CPX_MULTIOBJ_METHOD
    """See :macros:`CPX_MULTIOBJ_METHOD` in the C API."""

    priority = _const.CPX_MULTIOBJ_PRIORITY
    """See :macros:`CPX_MULTIOBJ_PRIORITY` in the C API."""

    blend = _const.CPX_MULTIOBJ_BLEND
    """See :macros:`CPX_MULTIOBJ_BLEND` in the C API."""

    pfeas = _const.CPX_MULTIOBJ_PFEAS
    """See :macros:`CPX_MULTIOBJ_PFEAS` in the C API."""

    dfeas = _const.CPX_MULTIOBJ_DFEAS
    """See :macros:`CPX_MULTIOBJ_DFEAS` in the C API."""


class MultiObjLongInfo(ConstantClass):
    """Types of long integer information that can be queried for
    multi-objective optimization.

    This class contains the types of solution information of type float
    that can be retreived from the solution of a sub-problem solved
    during multi-objective optimization.

    This information can be querried for each priority level with method
    `MultiObjSolnInterface.get_info`.
    """

    num_barrier_iterations = _const.CPX_MULTIOBJ_BARITCNT
    """See :macros:`CPX_MULTIOBJ_BARITCNT` in the C API."""

    num_sifting_iterations = _const.CPX_MULTIOBJ_SIFTITCNT
    """See :macros:`CPX_MULTIOBJ_SIFTITCNT` in the C API."""

    num_sifting_phase1_iterations = _const.CPX_MULTIOBJ_SIFTPHASE1CNT
    """See :macros:`CPX_MULTIOBJ_SIFTPHASE1CNT` in the C API."""

    num_degenerate_iterations = _const.CPX_MULTIOBJ_DEGCNT
    """See :macros:`CPX_MULTIOBJ_DEGCNT` in the C API."""

    num_iterations = _const.CPX_MULTIOBJ_ITCNT
    """See :macros:`CPX_MULTIOBJ_ITCNT` in the C API."""

    num_phase1_iterations = _const.CPX_MULTIOBJ_PHASE1CNT
    """See :macros:`CPX_MULTIOBJ_PHASE1CNT` in the C API."""

    num_primal_pushes = _const.CPX_MULTIOBJ_PPUSH
    """See :macros:`CPX_MULTIOBJ_PPUSH` in the C API."""

    num_primal_exchanges = _const.CPX_MULTIOBJ_PEXCH
    """See :macros:`CPX_MULTIOBJ_PEXCH` in the C API."""

    num_dual_pushes = _const.CPX_MULTIOBJ_DPUSH
    """See :macros:`CPX_MULTIOBJ_DPUSH` in the C API."""

    num_dual_exchanges = _const.CPX_MULTIOBJ_DEXCH
    """See :macros:`CPX_MULTIOBJ_DEXCH` in the C API."""

    num_nodes = _const.CPX_MULTIOBJ_NODECNT
    """See :macros:`CPX_MULTIOBJ_NODECNT` in the C API."""

    num_nodes_left = _const.CPX_MULTIOBJ_NODELEFTCNT
    """See :macros:`CPX_MULTIOBJ_NODELEFTCNT` in the C API."""


class MultiObjSolnInterface(BaseInterface):
    """Methods for accessing solutions for multi-objective models.

    The methods in this interface can be used to query solutions for
    multi-objective models.

    See also `MultiObjInterface` where methods for adding, querying, and
    modifying multiple objectives can be found.
    """

    float_info = MultiObjFloatInfo()
    """See `MultiObjFloatInfo()` """

    int_info = MultiObjIntInfo()
    """See `MultiObjIntInfo()` """

    long_info = MultiObjLongInfo()
    """See `MultiObjLongInfo()` """

    def __init__(self, parent):
        """Creates a new MIPSolutionInterface.

        The multi-objective solution interface is exposed by the
        top-level `Cplex` class as Cplex.solution.multiobj. This
        constructor is not meant to be used externally.
        """
        super().__init__(cplex=parent._cplex, advanced=True)

    def get_objective_value(self, objidx):
        """Returns the value of an objective function.

        objidx is the name or index of the objective to be accessed.

        See :cpxapi:`CPXmultiobjgetobjval` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.multiobj.set_num(2)
        >>> c.multiobj.set_linear(
        ...     1,
        ...     [(i, o) for i, o in enumerate(c.multiobj.get_linear(0))]
        ... )
        >>> c.solve()
        >>> c.solution.multiobj.get_objective_value(0)
        -202.5
        >>> c.solution.multiobj.get_objective_value(1)
        -202.5
        """
        objidx = self._cplex.multiobj._conv(objidx)
        return _proc.multiobjgetobjval(
            self._env._e,
            self._cplex._lp,
            objidx)

    def get_objval_by_priority(self, priority):
        """Returns the value of an objective function by priority.

        After multi-objective optimization, returns the blended objective
        value for the specified priority.

        See :cpxapi:`CPXmultiobjgetobjvalbypriority` in the Callable
        Library Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.multiobj.set_num(2)
        >>> c.multiobj.set_linear(
        ...     1,
        ...     [(i, o) for i, o in enumerate(c.multiobj.get_linear(0))]
        ... )
        >>> c.multiobj.set_priority(0, 1)
        >>> c.multiobj.set_priority(1, 2)
        >>> c.solve()
        >>> c.solution.multiobj.get_objval_by_priority(1)
        -202.5
        >>> c.solution.multiobj.get_objval_by_priority(2)
        -202.5
        """
        return _proc.multiobjgetobjvalbypriority(
            self._env._e,
            self._cplex._lp,
            priority)

    def get_num_solves(self):
        """Returns the number of sub-problems that where successfully
        solved during the last optimization of a multi-objective problem.

        See :cpxapi:`CPXmultiobjgetnumsolves` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.multiobj.set_num(2)
        >>> c.multiobj.set_linear(
        ...     1,
        ...     [(i, o) for i, o in enumerate(c.multiobj.get_linear(0))]
        ... )
        >>> c.multiobj.set_priority(0, 1)
        >>> c.multiobj.set_priority(1, 2)
        >>> c.solve()
        >>> c.solution.multiobj.get_num_solves()
        2
        """
        return _proc.multiobjgetnumsolves(self._env._e, self._cplex._lp)

    @staticmethod
    def _isintinfo(what):
        try:
            _ = MultiObjSolnInterface.int_info[what]
            return True
        except KeyError:
            return False

    @staticmethod
    def _islonginfo(what):
        try:
            _ = MultiObjSolnInterface.long_info[what]
            return True
        except KeyError:
            return False

    @staticmethod
    def _isdblinfo(what):
        try:
            _ = MultiObjSolnInterface.float_info[what]
            return True
        except KeyError:
            return False

    def get_info(self, subprob, what):
        """Returns the solution information of a sub-problem of a
        multi-objective optimization.

        subprob is the sub-problem of a multi-objective optimization.

        what is an attribute from Cplex.solution.multiobj.float_info
        (see `MultiObjFloatInfo`), Cplex.solution.multiobj.int_info
        (see `MultiObjIntInfo`), or Cplex.solution.multiobj.long_info
        (see `MultiObjLongInfo`).

        See :cpxapi:`CPXmultiobjgetdblinfo`,
        :cpxapi:`CPXmultiobjgetintinfo`,
        :cpxapi:`CPXmultiobjgetlonginfo`, etc. in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("example.mps")
        >>> c.multiobj.set_num(2)
        >>> c.multiobj.set_linear(
        ...     1,
        ...     [(i, o) for i, o in enumerate(c.multiobj.get_linear(0))]
        ... )
        >>> c.multiobj.set_priority(0, 1)
        >>> c.multiobj.set_priority(1, 2)
        >>> c.solve()
        >>> num = c.solution.multiobj.get_num_solves()
        >>> for i in range(num):
        ...     priority = c.solution.multiobj.get_info(
        ...         i,
        ...         c.solution.multiobj.int_info.priority
        ...     )
        """
        if MultiObjSolnInterface._isintinfo(what):
            return _proc.multiobjgetintinfo(self._env._e, self._cplex._lp,
                                            subprob, what)
        if MultiObjSolnInterface._islonginfo(what):
            return _proc.multiobjgetlonginfo(self._env._e, self._cplex._lp,
                                             subprob, what)
        if MultiObjSolnInterface._isdblinfo(what):
            return _proc.multiobjgetdblinfo(self._env._e, self._cplex._lp,
                                            subprob, what)
        raise ValueError(what)
