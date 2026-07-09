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
"""A module for the SolutionStrategy class."""
from . import _constantsenum
from ..constant_class import ConstantClass


class SolutionStrategy(ConstantClass):
    """The different types of solutions that can submitted to
    `cplex.callbacks.Context.post_heuristic_solution()`.

    For further details about these values, see the reference manual of
    the CPLEX Callable Library (C API) particularly, the enumeration
    :enum:`CPXCALLBACKSOLUTIONSTRATEGY`.
    """

    no_check = _constantsenum.CPXCALLBACKSOLUTION_NOCHECK
    """See CPXCALLBACKSOLUTION_NOCHECK in the C API."""

    check_feasible = _constantsenum.CPXCALLBACKSOLUTION_CHECKFEAS
    """See CPXCALLBACKSOLUTION_CHECKFEAS in the C API."""

    propagate = _constantsenum.CPXCALLBACKSOLUTION_PROPAGATE
    """See CPXCALLBACKSOLUTION_PROPAGATE in the C API."""

    solve = _constantsenum.CPXCALLBACKSOLUTION_SOLVE
    """See CPXCALLBACKSOLUTION_SOLVE in the C API."""
