# --------------------------------------------------------------------------
# Version 22.2.0
# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2000, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""A module for the CallbackInfo class."""
from . import _constantsenum
from ..constant_class import ConstantClass


class CallbackInfo(ConstantClass):
    """The values that can be used with
    `cplex.callbacks.Context.get_int_info()`,
    `cplex.callbacks.Context.get_long_info()`,
    and `cplex.callbacks.Context.get_double_info()`.

    See the reference manual of the CPLEX Callable Library (C API)
    for the constants in the enumeration :enum:`CPXCALLBACKINFO` for
    details about what those values query.
    """

    thread_id = _constantsenum.CPXCALLBACKINFO_THREADID
    """See CPXCALLBACKINFO_THREADID in the C API."""

    node_count = _constantsenum.CPXCALLBACKINFO_NODECOUNT
    """See CPXCALLBACKINFO_NODECOUNT in the C API."""

    iteration_count = _constantsenum.CPXCALLBACKINFO_ITCOUNT
    """See CPXCALLBACKINFO_ITCOUNT in the C API."""

    best_solution = _constantsenum.CPXCALLBACKINFO_BEST_SOL
    """See CPXCALLBACKINFO_BEST_SOL in the C API."""

    best_bound = _constantsenum.CPXCALLBACKINFO_BEST_BND
    """See CPXCALLBACKINFO_BEST_BND in the C API."""

    threads = _constantsenum.CPXCALLBACKINFO_THREADS
    """See CPXCALLBACKINFO_THREADS in the C API."""

    feasible = _constantsenum.CPXCALLBACKINFO_FEASIBLE
    """See CPXCALLBACKINFO_FEASIBLE in the C API."""

    time = _constantsenum.CPXCALLBACKINFO_TIME
    """See CPXCALLBACKINFO_TIME in the C API."""

    deterministic_time = _constantsenum.CPXCALLBACKINFO_DETTIME
    """See CPXCALLBACKINFO_DETTIME in the C API."""

    node_uid = _constantsenum.CPXCALLBACKINFO_NODEUID
    """See CPXCALLBACKINFO_NODEUID in the C API."""

    node_depth = _constantsenum.CPXCALLBACKINFO_NODEDEPTH
    """See CPXCALLBACKINFO_NODEDEPTH in the C API."""

    candidate_source = _constantsenum.CPXCALLBACKINFO_CANDIDATE_SOURCE
    """See CPXCALLBACKINFO_CANDIDATE_SOURCE in the C API."""

    restarts = _constantsenum.CPXCALLBACKINFO_RESTARTS
    """See CPXCALLBACKINFO_RESTARTS in the C API."""

    after_cut_loop = _constantsenum.CPXCALLBACKINFO_AFTERCUTLOOP
    """See CPXCALLBACKINFO_AFTERCUTLOOP in the C API."""

    nodes_left = _constantsenum.CPXCALLBACKINFO_NODESLEFT
    """See CPXCALLBACKINFO_NODESLEFT in the C API."""
