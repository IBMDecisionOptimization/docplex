# --------------------------------------------------------------------------
# File: __init__.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""The CPLEX Python API.

This package contains classes for accessing CPLEX from the Python
programming language. The most important class defined by this package
is the `Cplex` class, which provides methods for creating, modifying,
querying, or solving an optimization problem, and for querying aspects of
a solution.

The `exceptions` module defines the exception classes that are raised
during abnormal operation by the CPLEX Python API.

The `callbacks` module defines callback classes that can be used to alter
the behavior of the algorithms used by CPLEX.

The constant `infinity`, defined in the cplex package, should be used to
set infinite upper and lower bounds.

The classes `SparsePair` and `SparseTriple` are used as input and output
classes for sparse vector and sparse matrix output, respectively. See
the documentation for individual methods for details about the usage
of these classes.
"""
__all__ = ["Cplex", "Stats", "Aborter", "callbacks", "exceptions",
           "infinity", "ParameterSet", "SparsePair", "SparseTriple",
           "model_info"]
__version__ = "22.2.0.0"

from contextlib import closing
from io import BytesIO
import os
import weakref

from .aborter import Aborter
from . import callbacks
from . import model_info
from .exceptions import CplexError, WrongNumberOfArgumentsError
from ._internal import ProblemType, Environment
from ._internal._anno import (DoubleAnnotationInterface,
                              LongAnnotationInterface)
from ._internal._aux_functions import (init_list_args,
                                       validate_arg_lengths)
from ._internal._matrices import SparsePair, SparseTriple
from ._internal._subinterfaces import (AdvancedCplexInterface,
                                       ConflictInterface,
                                       FeasoptInterface,
                                       IndicatorConstraintInterface,
                                       InitialInterface,
                                       LinearConstraintInterface,
                                       MIPStartsInterface,
                                       ObjectiveInterface,
                                       ObjSense,
                                       OrderInterface,
                                       PresolveInterface,
                                       QuadraticConstraintInterface,
                                       SolutionInterface,
                                       SOSInterface,
                                       VariablesInterface)
from ._internal import _constants as _const
from ._internal import _procedural as _proc
from ._internal._multiobj import MultiObjInterface
from ._internal._parameter_classes import RootParameterGroup  # noqa: F401
from ._internal._pwl import PWLConstraintInterface
from .paramset import ParameterSet

infinity = _const.CPX_INFBOUND
"""See CPX_INFBOUND in the C API."""


class Stats():
    """A class whose data members reflect statistics about a CPLEX
    problem.

    An instance of this class is returned by the `Cplex.get_stats()`
    method.

    The __str__ method of this class returns a string containing a
    summary of the problem statistics in human readable form.

    An instance of this class always has the following integer members:

    * num_objectives
    * num_variables
    * num_nonnegative
    * num_fixed
    * num_boxed
    * num_free
    * num_other
    * num_binary
    * num_integer
    * num_semicontinuous
    * num_semiinteger
    * num_quadratic_variables
    * num_linear_objective_nz
    * num_quadratic_objective_nz
    * num_linear_constraints
    * num_linear_less
    * num_linear_equal
    * num_linear_greater
    * num_linear_range
    * num_linear_nz
    * num_linear_rhs_nz
    * num_indicator_constraints
    * num_indicator_less
    * num_indicator_equal
    * num_indicator_greater
    * num_indicator_complemented
    * num_indicator_nz
    * num_indicator_rhs_nz
    * num_quadratic_constraints
    * num_quadratic_less
    * num_quadratic_greater
    * num_quadratic_linear_nz
    * num_quadratic_nz
    * num_quadratic_rhs_nz
    * num_SOS_constraints
    * num_SOS1
    * num_SOS1_members
    * type_SOS1
    * num_SOS2
    * num_SOS2_members
    * type_SOS2
    * num_lazy_constraints
    * num_lazy_nnz
    * num_lazy_lt
    * num_lazy_eq
    * num_lazy_gt
    * num_lazy_rhs_nnz
    * num_user_cuts
    * num_user_cuts_nnz
    * num_user_cuts_lt
    * num_user_cuts_eq
    * num_user_cuts_gt
    * num_user_cuts_rhs_nnz
    * num_pwl_constraints
    * num_pwl_breaks

    An instance of this class always has the following float members:

    * min_lower_bound
    * max_upper_bound
    * min_linear_objective
    * max_linear_objective
    * min_linear_constraints
    * max_linear_constraints
    * min_linear_constraints_rhs
    * max_linear_constraints_rhs

    An instance of this class returned by an instance of the Cplex
    class with a quadratic objective also has the following float
    members:

    * min_quadratic_objective
    * max_quadratic_objective

    An instance of this class returned by an instance of the Cplex
    class with ranged constraints also has the following float
    members:

    * min_linear_range
    * max_linear_range

    An instance of this class returned by an instance of the Cplex
    class with quadratic constraints also has the following float
    members:

    * min_quadratic_linear
    * max_quadratic_linear
    * min_quadratic
    * max_quadratic
    * min_quadratic_rhs
    * max_quadratic_rhs

    An instance of this class returned by an instance of the Cplex
    class with indicator constraints also has the following float
    members:

    * min_indicator
    * max_indicator
    * min_indicator_rhs
    * max_indicator_rhs

    An instance of this class returned by an instance of the Cplex
    class with lazy constraints also has the following float members:

    * min_lazy_constraint
    * max_lazy_constraint
    * min_lazy_constraint_rhs
    * max_lazy_constraint_rhs

    An instance of this class returned by an instance of the Cplex
    class with user cuts also has the following float members:

    * min_user_cut
    * max_user_cut
    * min_user_cut_rhs
    * max_user_cut_rhs

    :undocumented: __init__
    """

    def __init__(self, c):
        self.name = c.get_problem_name()
        self.sense = c.objective.sense[c.objective.get_sense()].capitalize()

        raw_stats = _proc.getprobstats(c._env._e, c._lp)

        # multi-objective stats
        self.num_objectives = raw_stats.objs

        # counts of problem objects
        # variable data
        self.num_variables = raw_stats.cols
        self.num_nonnegative = raw_stats.ncnt
        self.num_fixed = raw_stats.xcnt
        self.num_boxed = raw_stats.bcnt
        self.num_free = raw_stats.fcnt
        self.num_other = raw_stats.ocnt
        self.num_binary = raw_stats.bicnt
        self.num_integer = raw_stats.icnt
        self.num_semicontinuous = raw_stats.scnt
        self.num_semiinteger = raw_stats.sicnt
        self.num_quadratic_variables = raw_stats.qpcnt
        self.num_linear_objective_nz = raw_stats.objcnt
        self.num_quadratic_objective_nz = raw_stats.qpnzcnt

        # linear constraint data
        self.num_linear_constraints = raw_stats.rows
        self.num_linear_less = raw_stats.lcnt
        self.num_linear_equal = raw_stats.ecnt
        self.num_linear_greater = raw_stats.gcnt
        self.num_linear_range = raw_stats.rngcnt
        self.num_linear_nz = raw_stats.nzcnt
        self.num_linear_rhs_nz = raw_stats.rhscnt

        # indicator data
        self.num_indicator_constraints = raw_stats.nindconstr
        self.num_indicator_less = raw_stats.indlcnt
        self.num_indicator_equal = raw_stats.indecnt
        self.num_indicator_greater = raw_stats.indgcnt
        self.num_indicator_complemented = raw_stats.indcompcnt
        self.num_indicator_nz = raw_stats.indnzcnt
        self.num_indicator_rhs_nz = raw_stats.indrhscnt

        # quadratic constraints
        self.num_quadratic_constraints = raw_stats.nqconstr
        self.num_quadratic_less = raw_stats.qlcnt
        self.num_quadratic_greater = raw_stats.qgcnt
        self.num_quadratic_linear_nz = raw_stats.linnzcnt
        self.num_quadratic_nz = raw_stats.quadnzcnt
        self.num_quadratic_rhs_nz = raw_stats.qrhscnt

        # SOS data
        self.num_SOS_constraints = raw_stats.nsos
        sos_string_list = ["",
                           "all continuous",
                           "all binary",
                           "all integer",
                           "continuous, binary, and integer",
                           "continuous and binary",
                           "continuous and integer",
                           "binary and integer", ]
        self.num_SOS1 = raw_stats.nsos1
        self.num_SOS1_members = raw_stats.sos1nmem
        self.type_SOS1 = sos_string_list[raw_stats.sos1type]
        self.num_SOS2 = raw_stats.nsos2
        self.num_SOS2_members = raw_stats.sos2nmem
        self.type_SOS2 = sos_string_list[raw_stats.sos2type]

        # lazy constraint data
        self.num_lazy_constraints = raw_stats.lazycnt
        self.num_lazy_nnz = raw_stats.lazynzcnt
        self.num_lazy_lt = raw_stats.lazylcnt
        self.num_lazy_eq = raw_stats.lazyecnt
        self.num_lazy_gt = raw_stats.lazygcnt
        self.num_lazy_rhs_nnz = raw_stats.lazyrhscnt

        # user cut data
        self.num_user_cuts = raw_stats.ucutcnt
        self.num_user_cuts_nnz = raw_stats.ucutnzcnt
        self.num_user_cuts_lt = raw_stats.ucutlcnt
        self.num_user_cuts_eq = raw_stats.ucutecnt
        self.num_user_cuts_gt = raw_stats.ucutgcnt
        self.num_user_cuts_rhs_nnz = raw_stats.ucutrhscnt

        # PWL constraints
        self.num_pwl_constraints = raw_stats.npwl
        self.num_pwl_breaks = raw_stats.npwlbreaks

        # min and max data
        # variables
        self.min_lower_bound = raw_stats.minlb
        self.max_upper_bound = raw_stats.maxub
        self.min_linear_objective = raw_stats.minobj
        self.max_linear_objective = raw_stats.maxobj
        if self.num_quadratic_objective_nz > 0:
            self.min_quadratic_objective = raw_stats.minqcoef
            self.max_quadratic_objective = raw_stats.maxqcoef

        # linear constraints
        self.min_linear_constraints = raw_stats.mincoef
        self.max_linear_constraints = raw_stats.maxcoef
        self.min_linear_constraints_rhs = raw_stats.minrhs
        self.max_linear_constraints_rhs = raw_stats.maxrhs
        if self.num_linear_range > 0:
            self.min_linear_range = raw_stats.minrng
            self.max_linear_range = raw_stats.maxrng

        # quadratic constraints
        if self.num_quadratic_constraints > 0:
            self.min_quadratic_linear = raw_stats.minqcl
            self.max_quadratic_linear = raw_stats.maxqcl
            self.min_quadratic = raw_stats.minqcq
            self.max_quadratic = raw_stats.maxqcq
            self.min_quadratic_rhs = raw_stats.minqcr
            self.max_quadratic_rhs = raw_stats.maxqcr

        # indicator constraints
        if self.num_indicator_constraints > 0:
            self.min_indicator = raw_stats.minind
            self.max_indicator = raw_stats.maxind
            self.min_indicator_rhs = raw_stats.minindrhs
            self.max_indicator_rhs = raw_stats.maxindrhs

        # lazy constraints
        if self.num_lazy_constraints > 0:
            self.min_lazy_constraint = raw_stats.minlazy
            self.max_lazy_constraint = raw_stats.maxlazy
            self.min_lazy_constraint_rhs = raw_stats.minlazyrhs
            self.max_lazy_constraint_rhs = raw_stats.maxlazyrhs

        # user cuts
        if self.num_user_cuts > 0:
            self.min_user_cut = raw_stats.minucut
            self.max_user_cut = raw_stats.maxucut
            self.min_user_cut_rhs = raw_stats.minucutrhs
            self.max_user_cut_rhs = raw_stats.maxucutrhs

    def __str__(self):
        """Returns a string containing a summary of the problem
        statistics in human readable form.
        """
        allinf = "all infinite"
        allzero = "all zero"
        sep = ",  "
        ret = ""
        ret = ret + "Problem name         : " + self.name + "\n"
        ret = ret + "Objective sense      : " + self.sense + "\n"
        ret = ret + "Variables            : %7d" % self.num_variables
        if self.num_nonnegative != self.num_variables or self.num_quadratic_variables > 0:
            ret = ret + "  ["
            sep_ind = 0
            if self.num_nonnegative > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Nneg: %d" % self.num_nonnegative
                sep_ind = 1
            if self.num_fixed > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Fix: %d" % self.num_fixed
                sep_ind = 1
            if self.num_boxed > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Box: %d" % self.num_boxed
                sep_ind = 1
            if self.num_free > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Free: %d" % self.num_free
                sep_ind = 1
            if self.num_binary > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Binary: %d" % self.num_binary
                sep_ind = 1
            if self.num_integer > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "General Integer: %d" % self.num_integer
                sep_ind = 1
            if self.num_semicontinuous > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Semi-continuous: %d" % self.num_semicontinuous
                sep_ind = 1
            if self.num_semiinteger > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Semi-integer: %d" % self.num_semiinteger
                sep_ind = 1
            if self.num_other > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Other: %d" % self.num_other
                sep_ind = 1
            if self.num_quadratic_variables > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Qobj: %d" % self.num_quadratic_variables
                sep_ind = 1
            ret = ret + "]"
        ret = ret + "\n"
        if self.num_objectives > 1:
            ret = ret + "Objectives           : %7d" % self.num_objectives + "\n"
            ret = ret + "  Objective nonzeros : %7d" % self.num_linear_objective_nz + "\n"
        else:
            ret = ret + "Objective nonzeros   : %7d" % self.num_linear_objective_nz + "\n"
        if self.num_quadratic_objective_nz > 0:
            ret = ret + "Objective Q nonzeros : %7d" % self.num_quadratic_objective_nz + "\n"
        ret = ret + "Linear constraints   : %7d" % self.num_linear_constraints
        if self.num_linear_constraints > 0:
            ret = ret + "  ["
            sep_ind = 0
            if self.num_linear_less > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Less: %d" % self.num_linear_less
                sep_ind = 1
            if self.num_linear_greater > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Greater: %d" % self.num_linear_greater
                sep_ind = 1
            if self.num_linear_equal > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Equal: %d" % self.num_linear_equal
                sep_ind = 1
            if self.num_linear_range > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Range: %d" % self.num_linear_range
                sep_ind = 1
            ret = ret + "]"
        ret = ret + "\n"
        ret = ret + "  Nonzeros           : %7d\n" % self.num_linear_nz
        ret = ret + "  RHS nonzeros       : %7d\n" % self.num_linear_rhs_nz
        if self.num_indicator_constraints > 0:
            ret = ret + \
                "Indicator constraints: %7d  [" % self.num_indicator_constraints
            sep_ind = 0
            if self.num_indicator_less > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Less: %d" % self.num_indicator_less
                sep_ind = 1
            if self.num_indicator_equal > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Equal: %d" % self.num_indicator_equal
                sep_ind = 1
            if self.num_indicator_greater > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Greater: %d" % self.num_indicator_greater
                sep_ind = 1
            ret = ret + "]\n"
            if self.num_indicator_complemented:
                ret = ret + "  Complemented       : %7d\n" % self.num_indicator_complemented
                ret = ret + "  Nonzeros           : %7d\n" % self.num_indicator_nz
                ret = ret + "  RHS nonzeros       : %7d\n" % self.num_indicator_rhs_nz
        if self.num_quadratic_constraints > 0:
            ret = ret + \
                "Quadratic constraints: %7d  [" % self.num_quadratic_constraints
            sep_ind = 0
            if self.num_quadratic_less > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Less: %d" % self.num_quadratic_less
                sep_ind = 1
            if self.num_quadratic_greater > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "Greater: %d" % self.num_quadratic_greater
                sep_ind = 1
            ret = ret + "]\n"
            ret = ret + "  Linear terms       : %7d\n" % self.num_quadratic_linear_nz
            ret = ret + "  Quadratic terms    : %7d\n" % self.num_quadratic_nz
            ret = ret + "  RHS nonzeros       : %7d\n" % self.num_quadratic_rhs_nz
        if self.num_SOS_constraints > 0:
            ret = ret + \
                "SOS                  : %7d  [" % self.num_SOS_constraints
            sep_ind = 0
            if self.num_SOS1 > 0:
                if sep_ind:
                    ret = ret + sep
                ret = ret + "SOS1: %d, %d members" % (self.num_SOS1,
                                                      self.num_SOS1_members)
                if self.type_SOS1:
                    ret += ", %s" % self.type_SOS1
                sep_ind = 1
            if self.num_SOS2 > 0:
                if sep_ind:
                    ret = ret + ";  "
                ret = ret + "SOS2: %d, %d members" % (self.num_SOS2,
                                                      self.num_SOS2_members)
                if self.type_SOS2:
                    ret += ", %s" % self.type_SOS2
                sep_ind = 1
            ret = ret + "]\n"
        if self.num_pwl_constraints > 0:
            ret = ret + \
                "PWL                  : %7d  [" % self.num_pwl_constraints
            if self.num_pwl_breaks > 0:
                ret = ret + "Breaks: %d" % self.num_pwl_breaks
            ret = ret + "]\n"
        ret = ret + "\n"
        if self.min_lower_bound > -infinity:
            valstr1 = str("%#-15.7g" % self.min_lower_bound)
        else:
            valstr1 = allinf
        if self.max_upper_bound < infinity:
            valstr2 = str("%#-15.7g" % self.max_upper_bound)
        else:
            valstr2 = allinf
        ret = ret + \
            "Variables            : Min LB: %-15s  Max UB: %-15s\n" % (
                valstr1, valstr2)
        if self.min_linear_objective > -infinity:
            valstr1 = str("%#-15.7g" % self.min_linear_objective)
        else:
            valstr1 = allzero
        if self.max_linear_objective < infinity:
            valstr2 = str("%#-15.7g" % self.max_linear_objective)
        else:
            valstr2 = allzero
        ret = ret + \
            "Objective nonzeros   : Min   : %-15s  Max   : %-15s\n" % (
                valstr1, valstr2)
        if self.num_quadratic_objective_nz > 0:
            if self.min_quadratic_objective > -infinity:
                valstr1 = str("%#-15.7g" % self.min_quadratic_objective)
            else:
                valstr1 = allzero
            if self.max_quadratic_objective < infinity:
                valstr2 = str("%#-15.7g" % self.max_quadratic_objective)
            else:
                valstr2 = allzero
            ret = ret + \
                "Objective Q nonzeros : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
        ret = ret + "Linear constraints   :\n"
        if self.min_linear_constraints > -infinity:
            valstr1 = str("%#-15.7g" % self.min_linear_constraints)
        else:
            valstr1 = allzero
        if self.max_linear_constraints < infinity:
            valstr2 = str("%#-15.7g" % self.max_linear_constraints)
        else:
            valstr2 = allzero
        ret = ret + \
            "  Nonzeros           : Min   : %-15s  Max   : %-15s\n" % (
                valstr1, valstr2)
        if self.min_linear_constraints_rhs > -infinity:
            valstr1 = str("%#-15.7g" % self.min_linear_constraints_rhs)
        else:
            valstr1 = allzero
        if self.max_linear_constraints_rhs < infinity:
            valstr2 = str("%#-15.7g" % self.max_linear_constraints_rhs)
        else:
            valstr2 = allzero
        ret = ret + \
            "  RHS nonzeros       : Min   : %-15s  Max   : %-15s\n" % (
                valstr1, valstr2)
        if self.num_linear_range > 0:
            ret = ret + "  Range values       : Min   : %#-15.7g  Max   : %#-15.7g\n" % (
                self.min_linear_range, self.max_linear_range)
        if self.num_quadratic_constraints > 0:
            ret = ret + "Quadratic constraints:\n"
            if self.min_quadratic_linear > -infinity:
                valstr1 = str("%#-15.7g" % self.min_quadratic_linear)
            else:
                valstr1 = allzero
            if self.max_quadratic_linear < infinity:
                valstr2 = str("%#-15.7g" % self.max_quadratic_linear)
            else:
                valstr2 = allzero
            ret = ret + \
                "  Linear terms       : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
            if self.min_quadratic > -infinity:
                valstr1 = str("%#-15.7g" % self.min_quadratic)
            else:
                valstr1 = allzero
            if self.max_quadratic < infinity:
                valstr2 = str("%#-15.7g" % self.max_quadratic)
            else:
                valstr2 = allzero
            ret = ret + \
                "  Quadratic terms    : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
            if self.min_quadratic_rhs > -infinity:
                valstr1 = str("%#-15.7g" % self.min_quadratic_rhs)
            else:
                valstr1 = allzero
            if self.max_quadratic_rhs < infinity:
                valstr2 = str("%#-15.7g" % self.max_quadratic_rhs)
            else:
                valstr2 = allzero
            ret = ret + \
                "  RHS nonzeros       : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
        if self.num_indicator_constraints > 0:
            ret = ret + "Indicator constraints:\n"
            if self.min_indicator > -infinity:
                valstr1 = str("%#-15.7g" % self.min_indicator)
            else:
                valstr1 = allzero
            if self.max_indicator < infinity:
                valstr2 = str("%#-15.7g" % self.max_indicator)
            else:
                valstr2 = allzero
            ret = ret + \
                "  Nonzeros           : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
            if self.min_indicator_rhs > -infinity:
                valstr1 = str("%#-15.7g" % self.min_indicator_rhs)
            else:
                valstr1 = allzero
            if self.max_indicator_rhs < infinity:
                valstr2 = str("%#-15.7g" % self.max_indicator_rhs)
            else:
                valstr2 = allzero
            ret = ret + \
                "  RHS nonzeros       : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
        if self.num_lazy_constraints > 0:
            ret = ret + "Lazy constraints     :\n"
            if self.min_lazy_constraint > -infinity:
                valstr1 = str("%#-15.7g" % self.min_lazy_constraint)
            else:
                valstr1 = allzero
            if self.max_lazy_constraint < infinity:
                valstr2 = str("%#-15.7g" % self.max_lazy_constraint)
            else:
                valstr2 = allzero
            ret = ret + \
                "  Nonzeros           : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
            if self.min_lazy_constraint_rhs > -infinity:
                valstr1 = str("%#-15.7g" % self.min_lazy_constraint_rhs)
            else:
                valstr1 = allzero
            if self.max_lazy_constraint_rhs < infinity:
                valstr2 = str("%#-15.7g" % self.max_lazy_constraint_rhs)
            else:
                valstr2 = allzero
            ret = ret + \
                "  RHS nonzeros       : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
        if self.num_user_cuts > 0:
            ret = ret + "User cuts            :\n"
            if self.min_user_cut > -infinity:
                valstr1 = str("%#-15.7g" % self.min_user_cut)
            else:
                valstr1 = allzero
            if self.max_user_cut < infinity:
                valstr2 = str("%#-15.7g" % self.max_user_cut)
            else:
                valstr2 = allzero
            ret = ret + \
                "  Nonzeros           : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
            if self.min_user_cut_rhs > -infinity:
                valstr1 = str("%#-15.7g" % self.min_user_cut_rhs)
            else:
                valstr1 = allzero
            if self.max_user_cut_rhs < infinity:
                valstr2 = str("%#-15.7g" % self.max_user_cut_rhs)
            else:
                valstr2 = allzero
            ret = ret + \
                "  RHS nonzeros       : Min   : %-15s  Max   : %-15s\n" % (
                    valstr1, valstr2)
        return ret


def _is_special_filetype(filename, filetype, ext):
    if filetype is None or filetype == "":
        for extra_ext in ('', '.gz', '.bz2'):
            if (isinstance(filename, str) and
                    filename.endswith('.' + ext + extra_ext)):
                return True
    else:
        if filetype == ext:
            return True
    return False


def _getcplexstudiodir():
    version, release, modification, _ = __version__.split(".")
    # There is a small oddity in how the environment variable
    # CPLEX_STUDIO_DIR<VERSION> is called: A "0" in MODIFICATION is
    # ignored (i.e., instead of "1280" for 12.8.0, we use "128").
    if modification == "0":
        modification = ""
    return "CPLEX_STUDIO_DIR{0}{1}{2}".format(version, release, modification)


def _setcpxchecklicdir():
    # Only try to set CPLEX_CPXCHECKLIC_BINDIR if CPLEX_STUDIO_KEY is set
    # and CPLEX_STUDIO_DIR and CPLEX_CPXCHECKLIC_BINDIR are not set.
    if ((not os.getenv("CPLEX_STUDIO_KEY")) or os.getenv(_getcplexstudiodir()) or os.getenv("CPLEX_CPXCHECKLIC_BINDIR")):
        return
    # First, try using shutil.which().
    import shutil
    cpxchecklic = shutil.which("cpxchecklic")
    if cpxchecklic:
        os.environ["CPLEX_CPXCHECKLIC_BINDIR"] = os.path.dirname(cpxchecklic)
        return
    # Then, if not successful, try using site.getusersitepackages().
    import site
    pydir = os.path.dirname(site.getusersitepackages())
    for dirname in ("bin", "Scripts"):
        cpxchecklic = os.path.join(pydir, dirname, "cpxchecklic")
        if os.path.exists(cpxchecklic):
            os.environ["CPLEX_CPXCHECKLIC_BINDIR"] = os.path.dirname(cpxchecklic)
            return
    # If neither of those works, then give up.


# Attempt to set CPLEX_CPXCHECKLIC_BINDIR automatically. This should
# happen once on "import cplex".
_setcpxchecklicdir()


class Cplex():
    """A class encapsulating a CPLEX Problem.

    An instance of the Cplex class provides methods for creating,
    modifying, and querying an optimization problem, solving it, and
    querying aspects of the solution.

    Most of the methods are provided within subinterfaces: for example,
    methods for adding, modifying, and querying data associated with
    variables are within the `Cplex.variables` interface, and methods for
    querying the solution are within the `Cplex.solution` category.

    :undocumented: __del__
    """

    problem_type = ProblemType()
    """See `ProblemType` """

    def __init__(self, *args):
        """Constructor of the Cplex class.

        The Cplex constructor accepts four types of argument lists.

        >>> cpx = cplex.Cplex()  # doctest: +SKIP

        cpx is a new problem with no data

        >>> cpx = cplex.Cplex("filename")  # doctest: +SKIP

        cpx is a new problem containing the data in filename. If filename
        does not exist, an exception is raised.

        >>> cpx = cplex.Cplex("filename", "filetype")  # doctest: +SKIP

        same as form 2, but cplex reads the file filename as a file of
        type filetype, rather than inferring the file type from its
        extension.

        >>> cpx = cplex.Cplex(old_cpx)  # doctest: +SKIP

        cpx contains the same problem data as old_cpx, but is a different
        object and contains no solution data. Future changes to one do
        not affect the other.

        The Cplex object is a context manager and can be used, like so:

        >>> import cplex
        >>> with cplex.Cplex() as cpx:
        ...     # do stuff
        ...     pass

        When the with-block is finished, the `end()` method will be
        called automatically.
        """
        # Declare and initialize attributes
        self._disposed = False
        self._aborter = None
        self._env = None
        self._lp = None
        self._pslst = []
        # Initialize data strucutures associated with CPLEX
        nargs = len(args)
        if nargs > 2:
            raise WrongNumberOfArgumentsError()
        self._env = Environment()
        if nargs == 0:
            self._lp = _proc.createprob(self._env._e, "")
        elif nargs == 1:
            if isinstance(args[0], Cplex):
                self._lp = _proc.cloneprob(self._env._e, args[0]._lp)
            elif isinstance(args[0], str):
                self._lp = _proc.createprob(self._env._e, args[0])
                _proc.readcopyprob(self._env._e, self._lp, args[0])
            else:
                raise TypeError("invalid argument: {0}".format(args[0]))
        else:
            assert nargs == 2
            if isinstance(args[0], str) and isinstance(args[1], str):
                self._lp = _proc.createprob(self._env._e, args[0])
                _proc.readcopyprob(self._env._e, self._lp, args[0], args[1])
            else:
                raise TypeError("invalid arguments: {0}".format(args))
        self._env_lp_ptr = _proc.pack_env_lp_ptr(self._env._e, self._lp)

        self.parameters = self._env.parameters
        """See `RootParameterGroup`"""
        self.parameters._cplex = weakref.proxy(self)

        self.variables = VariablesInterface(self)
        """See `VariablesInterface`"""

        self.linear_constraints = LinearConstraintInterface(
            self)
        """See `LinearConstraintInterface`"""

        self.quadratic_constraints = QuadraticConstraintInterface(self)
        """See `QuadraticConstraintInterface`"""

        self.indicator_constraints = IndicatorConstraintInterface(self)
        """See `IndicatorConstraintInterface`"""

        self.SOS = SOSInterface(self)
        """See `SOSInterface`"""

        self.objective = ObjectiveInterface(self)
        """See `ObjectiveInterface`"""

        self.multiobj = MultiObjInterface(self)
        """See `MultiObjInterface`"""

        self.MIP_starts = MIPStartsInterface(self)
        """See `MIPStartsInterface`"""

        self.solution = SolutionInterface(self)
        """See `SolutionInterface`"""

        self.presolve = PresolveInterface(self)
        """See `PresolveInterface`"""

        self.order = OrderInterface(self)
        """See `OrderInterface`"""

        self.conflict = ConflictInterface(self)
        """See `ConflictInterface`"""

        self.advanced = AdvancedCplexInterface(self)
        """See `AdvancedCplexInterface`"""

        self.start = InitialInterface(self)
        """See `InitialInterface`"""

        self.feasopt = FeasoptInterface(self)
        """See `FeasoptInterface`"""

        self.long_annotations = LongAnnotationInterface(self)
        """See `LongAnnotationInterface`"""

        self.double_annotations = DoubleAnnotationInterface(self)
        """See `DoubleAnnotationInterface`"""

        self.pwl_constraints = PWLConstraintInterface(self)
        """See `PWLConstraintInterface`"""

    def end(self):
        """Releases the Cplex object.

        Frees all data structures associated with CPLEX. After a call of
        the method end(), the invoking Cplex object and all objects that
        have been created with it (such as variables and constraints) can
        no longer be used. Attempts to use them subsequently raise a
        ValueError.

        Note
          The Cplex object is a context manager. Thus, rather than
          calling this method explicitly, the best practice should be to
          use a Cplex object in a "with" statement (see `__enter__` and
          `__exit__`).

        Example usage:

        >>> import cplex
        >>> cpx = cplex.Cplex()
        >>> cpx.end()
        """
        if self._disposed:
            return
        self._disposed = True
        # free aborter if necc.
        if self._aborter:
            self.remove_aborter()
        # free prob
        # Note: self._env._e may already be None if the Environment object
        # was garbage collected before this Cplex object (Python 3.13+
        # incremental GC may collect objects in a different order).
        if self._env and self._lp and self._env._e:
            try:
                _proc.setgenericcallbackfunc(self._env._e, self._lp, 0, None)
            except:  # noqa: E722
                # Ignore exception in destructor, in particular we may
                # get CPXERR_NOT_ONE_PROBLEM here.
                pass
            try:
                _proc.freeprob(self._env._e, self._lp)
            except ValueError:
                # self._env._e may have been freed already (e.g., if
                # Environment.__del__ ran before Cplex.__del__).
                pass
        # free parameter sets if necc.
        for ps in self._pslst:
            ps.end()
        # free env
        if self._env:
            self._env._end()

    def __del__(self):
        """non-public"""
        self.end()

    def __enter__(self):
        """Enter the runtime context related to this object.

        The "with" statement will bind this method's return value to the
        target specified in the as clause of the statement, if any.

        Cplex objects return themselves.

        Example usage:

        >>> import cplex
        >>> with cplex.Cplex() as cpx:
        ...     # do stuff
        ...     pass
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context.

        When we exit the with-block, the `end()` method is called
        automatically.
        """
        self.end()

    def read(self, filename, filetype=""):
        """Reads a problem from file.

        The first argument is a string specifying the filename from which
        the problem will be read.

        If the method is called with two arguments, the second argument
        is a string specifying the file type. If this argument is
        omitted, filetype is taken to be the extension of the filename.

        See :cpxapi:`CPXreadcopyprob` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        """
        _proc.readcopyprob(self._env._e, self._lp, filename, filetype)

    def write(self, filename, filetype=""):
        """Writes a problem to a file.

        The first argument is a string specifying the filename to which
        the problem will be written.

        If the filename ends with .bz2 (for BZip2) or .gz (for GNU Zip),
        a compressed file is written.

        If the method is called with two arguments, the second argument
        is a string specifying the file type. If this argument is
        omitted, filetype is taken to be the extension of the filename.

        If filetype is any of "sav", "mps", "lp", the problem is written
        in the corresponding format. If filetype is either "rew" or "rlp"
        the problem is written with generic names in mps or lp format,
        respectively. If filetype is "alp" the problem is written with
        generic names in lp format, where the variable names are
        annotated to indicate the type and bounds of each variable.

        If filetype is "dua", the dual problem is written to a file. If
        filetype is "emb", an embedded network problem is written to a
        file. If filetype is "ppe", the perturbed problem is written to a
        file. If filetype is "dpe", the perturbed dual problem is written
        to a file.

        For documentation of the file types, see the CPLEX File Format
        Reference Manual.

        See :cpxapi:`CPXwriteprob`, :cpxapi:`CPXdualwrite`,
        :cpxapi:`CPXembwrite`, :cpxapi:`CPXdperwrite`, and
        :cpxapi:`CPXpperwrite` in the Callable Library Reference Manual
        for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(names=['x1', 'x2', 'x3'])
        >>> c.write("example.lp")
        """
        if _is_special_filetype(filename, filetype, 'dua'):
            _proc.dualwrite(self._env._e, self._lp, filename)
        elif _is_special_filetype(filename, filetype, 'emb'):
            _proc.embwrite(self._env._e, self._lp, filename)
        elif _is_special_filetype(filename, filetype, 'dpe'):
            epsilon = self.parameters.simplex.perturbation.constant.get()
            _proc.dperwrite(self._env._e, self._lp, filename, epsilon)
        elif _is_special_filetype(filename, filetype, 'ppe'):
            epsilon = self.parameters.simplex.perturbation.constant.get()
            _proc.pperwrite(self._env._e, self._lp, filename, epsilon)
        else:
            _proc.writeprob(self._env._e, self._lp, filename, filetype)

    def write_to_stream(self, stream, filetype='LP', comptype=''):
        """Writes a problem to a file-like object in the given file format.

        The filetype argument can be any of "sav" (a binary format), "lp"
        (the default), "mps", "rew", "rlp", or "alp" (see `Cplex.write`
        for an explanation of these).

        If comptype is "bz2" (for BZip2) or "gz" (for GNU Zip), a
        compressed file is written.

        See :cpxapi:`CPXwriteprob` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(names=['x1', 'x2', 'x3'])
        >>> class NoOpStream():
        ...     def __init__(self):
        ...         self.was_called = False
        ...     def write(self, bytes):
        ...         self.was_called = True
        ...         pass
        ...     def flush(self):
        ...         pass
        >>> stream = NoOpStream()
        >>> c.write_to_stream(stream)
        >>> stream.was_called
        True
        """
        try:
            callable(stream.write)
        except AttributeError:
            raise CplexError("stream must have a write method")
        try:
            callable(stream.flush)
        except AttributeError:
            raise CplexError("stream must have a flush method")
        # Since there is no filename argument, we validate the
        # compression type.
        if comptype not in ('', 'bz2', 'gz'):
            raise ValueError(
                "invalid compression type specified for comptype: {0}".format(
                    comptype))
        # Any base name will do for the filename. Note that the
        # compression type must be specified in the filename (not the
        # filetype).
        filename = "prob.{0}".format(filetype)
        if comptype:
            filename += ".{0}".format(comptype)
        _proc.writeprobdev(self._env._e, self._lp, stream,
                           filename, filetype)

    def write_as_string(self, filetype='LP', comptype=''):
        """Writes a problem as a string in the given file format.

        For an explanation of the filetype and comptype arguments, see
        `Cplex.write_to_stream`.

        Note
          When SAV format is specified for filetype or a compressed file
          format is specified for comptype, the return value will be a
          byte string.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(names=['x1', 'x2', 'x3'])
        >>> lp_str = c.write_as_string("lp")
        >>> len(lp_str) > 0
        True
        """
        fileenc = self.parameters.read.fileencoding.get()
        with closing(BytesIO()) as output:
            self.write_to_stream(output, filetype, comptype)
            result = output.getvalue()
            # Never decode for SAV format nor compressed files.
            if not (filetype.lower().startswith("sav") or comptype):
                result = result.decode(fileenc)
            return result

    def read_annotations(self, filename):
        """Reads annotations from a file.

        See :cpxapi:`CPXreadcopyannotations` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> idx = c.long_annotations.add('ann1', 0)
        >>> objtype = c.long_annotations.object_type.variable
        >>> indices = c.variables.add(names=['v1', 'v2', 'v3'])
        >>> c.long_annotations.set_values(idx, objtype,
        ...                               [(i, 1) for i in indices])
        >>> idx = c.double_annotations.add('ann1', 0)
        >>> objtype = c.double_annotations.object_type.variable
        >>> indices = c.variables.add(names=['v1', 'v2', 'v3'])
        >>> c.double_annotations.set_values(idx, objtype,
        ...                                 [(i, 1) for i in indices])
        >>> c.write_annotations('example.ann')
        >>> c.long_annotations.delete()
        >>> c.double_annotations.delete()
        >>> c.long_annotations.get_num()
        0
        >>> c.double_annotations.get_num()
        0
        >>> c.read_annotations('example.ann')
        >>> c.long_annotations.get_num()
        1
        >>> c.double_annotations.get_num()
        1
        """
        _proc.readcopyanno(self._env._e, self._lp, filename)

    def write_annotations(self, filename):
        """Writes the annotations to a file.

        See :cpxapi:`CPXwriteannotations` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> idx = c.long_annotations.add('ann1', 0)
        >>> objtype = c.long_annotations.object_type.variable
        >>> indices = c.variables.add(names=['v1', 'v2', 'v3'])
        >>> c.long_annotations.set_values(idx, objtype,
        ...                               [(i, 1) for i in indices])
        >>> c.write_annotations('example.ann')
        """
        _proc.writeanno(self._env._e, self._lp, filename)

    def write_benders_annotation(self, filename):
        """Writes the annotation of the auto-generated decomposition.

        Writes the annotation of the decompostion CPLEX automatically
        generates for the model of the CPLEX problem object to the
        specified file.

        See :cpxapi:`CPXwritebendersannotation` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read('UFL_25_35_1.mps')
        >>> c.write_benders_annotation('UFL_25_35_1.ann')
        """
        _proc.writebendersanno(self._env._e, self._lp, filename)

    def get_problem_type(self):
        """Returns the problem type.

        See :cpxapi:`CPXgetprobtype` in the Callable Library Reference
        Manual for more detail.

        The return value is an attribute of `problem_type`.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.get_problem_type()
        0
        >>> c.problem_type[c.get_problem_type()]
        'LP'
        """
        return _proc.getprobtype(self._env._e, self._lp)

    def set_problem_type(self, type, soln=None):
        """Changes the problem type.

        If only one argument is given, that argument specifies the new
        problem type (see `problem_type`). It must be one of the
        following:

        * Cplex.problem_type.LP
        * Cplex.problem_type.MILP
        * Cplex.problem_type.fixed_MILP
        * Cplex.problem_type.QP
        * Cplex.problem_type.MIQP
        * Cplex.problem_type.fixed_MIQP
        * Cplex.problem_type.QCP
        * Cplex.problem_type.MIQCP

        If an optional second argument is given, it is taken to be an
        identifier of a member of the solution pool. In this case, the
        first argument must be one of the following:

        * Cplex.problem_type.fixed_MILP
        * Cplex.problem_type.fixed_MIQP

        See :cpxapi:`CPXchgprobtype` and :cpxapi:`CPXchgprobtypesolnpool`
        in the Callable Library Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.set_problem_type(c.problem_type.LP)
        """
        if soln is None:
            _proc.chgprobtype(self._env._e, self._lp, type)
        else:
            _proc.chgprobtypesolnpool(self._env._e, self._lp, type, soln)

    def _is_MIP(self):
        """non-public"""
        probtype = self.get_problem_type()
        return probtype in (Cplex.problem_type.MILP,
                            Cplex.problem_type.MIQP,
                            Cplex.problem_type.MIQCP)

    def _setup_callbacks(self):
        """non-public"""
        for cb in self._env._callbacks:
            cb._env_lp_ptr = self._env_lp_ptr
            if hasattr(cb, "_setup"):
                cb._setup(self._env._e, self._lp)

    def solve(self, paramsets=None):
        """Solves the problem.

        The optional paramsets argument can only be specified when
        multiple objectives are present (otherwise, a ValueError is
        raised). paramsets must be a sequence containing `ParameterSet`
        objects (see `Cplex.create_parameter_set`) or None. See
        :cpxapi:`CPXmultiobjopt` in the Callable Library Reference Manual
        for more detail.

        Note
          The solve method returning normally (i.e., without raising an
          exception) does not necessarily mean that an optimal or
          feasible solution has been found. Use
          `SolutionInterface.get_status()` to query the status of the
          current solution.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> c.solve()
        >>> status = c.solution.get_status()
        """
        (paramsets,) = init_list_args(paramsets)
        self._setup_callbacks()
        ismultiobj = _proc.ismultiobj(self._env._e, self._lp)
        if (not ismultiobj and paramsets):
            raise ValueError("paramsets argument can only be specified"
                             " for a multi-objective model")
        if ismultiobj:
            nprios = _proc.getnumprios(self._env._e, self._lp)
            nparamsets = len(paramsets)
            if nparamsets > 0 and nprios != nparamsets:
                raise ValueError("if specified, len(paramsets) ({0})"
                                 " must be equal to the number of"
                                 " priorities ({1})".format(nparamsets, nprios))
            _proc.multiobjopt(self._env._e, self._lp,
                              [None if ps is None else ps._ps
                               for ps in paramsets])
        elif self._is_MIP():
            _proc.mipopt(self._env._e, self._lp)
        elif self.quadratic_constraints.get_num() > 0:
            lpmethod = self.parameters.lpmethod.get()
            if lpmethod in (_const.CPX_ALG_BARRIER, _const.CPX_ALG_AUTOMATIC):
                _proc.hybbaropt(self._env._e, self._lp, _const.CPX_ALG_NONE)
            else:
                _proc.qpopt(self._env._e, self._lp)
        elif not self.objective.get_num_quadratic_nonzeros() > 0:
            _proc.lpopt(self._env._e, self._lp)
        else:
            _proc.qpopt(self._env._e, self._lp)

    def runseeds(self, cnt=30):
        """Evaluates the variability of the problem.

        Solves the same problem instance multiple times using different
        random seeds allowing the user to evaluate the variability of the
        problem class the instance belongs to.

        The optional cnt argument specifies the number of times
        optimization should be performed (the default is 30).

        A problem must be an MILP, MIQP, or MIQCP and must exist in
        memory.
        """
        self._setup_callbacks()
        _proc.runseeds(self._env._e, self._lp, cnt)

    def populate_solution_pool(self):
        """Generates a variety of solutions to a discrete problem (MIP, MIQP, MIQCP).

        The algorithm that populates the solution pool works in two
        phases.

        In the first phase, it solves the problem to optimality (or
        some stopping criterion set by the user) while it sets up a
        branch and cut tree for the second phase.

        In the second phase, it generates multiple solutions by using
        the information computed and stored in the first phase and by
        continuing to explore the tree.

        For more information, see the function :mipapi:`CPXpopulate` in the
        Callable Library Reference Manual and the topic solution pool
        in the CPLEX User's Manual.
        """
        self._setup_callbacks()
        _proc.populate(self._env._e, self._lp)

    def get_problem_name(self):
        """Returns the problem name.

        See :cpxapi:`CPXgetprobname` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.set_problem_name("prob1")
        >>> c.get_problem_name()
        'prob1'
        """
        return _proc.getprobname(self._env._e, self._lp)

    def set_problem_name(self, name):
        """Sets the problem name.

        See :cpxapi:`CPXchgprobname` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.set_problem_name("prob1")
        >>> c.get_problem_name()
        'prob1'
        """
        _proc.chgprobname(self._env._e, self._lp, name)

    def cleanup(self, epsilon):
        """Deletes values from the problem data with absolute value
        smaller than epsilon.

        See :cpxapi:`CPXcleanup` in the Callable Library Reference Manual
        for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> indices = c.variables.add(obj=[1.0, 1e-10, 1.0])
        >>> c.objective.get_linear()
        [1.0, 1e-10, 1.0]
        >>> c.cleanup(epsilon=1e-6)
        >>> c.objective.get_linear()
        [1.0, 0.0, 1.0]
        """
        _proc.cleanup(self._env._e, self._lp, epsilon)

    def register_callback(self, callback_class):
        """Registers a callback class for use during optimization.

        callback_class must be a proper subclass of one of the callback
        classes defined in the module `callbacks`. To implement custom
        logic, override the __call__ method with a method that has
        signature __call__(self) -> None. If callback_class is a subclass
        of more than one callback class, it will only be called when its
        first superclass is called. register_callback returns the
        instance of callback_class registered for use. Any previously
        registered callback of the same class will no longer be
        registered.

        Returns an instance of callback_class.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> class MyMIPInfoCallback(cplex.callbacks.MIPInfoCallback):
        ...     pass
        >>> cb = c.register_callback(MyMIPInfoCallback)
        """
        return self._env.register_callback(callback_class)

    def unregister_callback(self, callback_class):
        """Stops a callback class from being used.

        callback_class must be one of the callback classes defined in the
        module `callbacks` or a subclass of one of them. This method
        unregisters any previously registered callback of the same class.
        If callback_class is a subclass of more than one callback class,
        this method will unregister only the callback of the same type as
        its first superclass.

        Returns the instance of callback_class just unregistered.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> class MyMIPInfoCallback(cplex.callbacks.MIPInfoCallback):
        ...     pass
        >>> cb = c.register_callback(MyMIPInfoCallback)
        >>> cb = c.unregister_callback(MyMIPInfoCallback)
        """
        return self._env.unregister_callback(callback_class)

    def set_results_stream(self, results_file, fn=None):
        """Specifies where results will be printed.

        The first argument must be a file-like object (i.e., an object
        with a write method and a flush method). Use None as the first
        argument to suppress output.

        The second optional argument is a function that takes a string as
        input and returns a string. If specified, strings sent to this
        stream will be processed by this function before being written.

        Returns the stream to which results will be written. To write to
        this stream, use this object's write() method.

        Example usage:

        >>> import cplex
        >>> with cplex.Cplex() as c, open("output.txt", "w") as f:
        ...     output = c.set_results_stream(f)
        ...     output.write("this is an example")
        """
        return self._env.set_results_stream(results_file, fn)

    def set_warning_stream(self, warning_file, fn=None):
        """Specifies where warnings will be printed.

        The first argument must be a file-like object (i.e., an object
        with a write method and a flush method). Use None as the first
        argument to suppress output.

        The second optional argument is a function that takes a string as
        input and returns a string. If specified, strings sent to this
        stream will be processed by this function before being written.

        Returns the stream to which warnings will be written. To write to
        this stream, use this object's write() method.

        Example usage:

        >>> import cplex
        >>> with cplex.Cplex() as c, open("output.txt", "w") as f:
        ...     output = c.set_warning_stream(f)
        ...     output.write("this is an example")
        """
        return self._env.set_warning_stream(warning_file, fn)

    def set_error_stream(self, error_file, fn=None):
        """Specifies where errors will be printed.

        The first argument must be a file-like object (i.e., an object
        with a write method and a flush method). Use None as the first
        argument to suppress output.

        The second optional argument is a function that takes a string as
        input and returns a string. If specified, strings sent to this
        stream will be processed by this function before being written.

        Returns the stream to which errors will be written. To write to
        this stream, use this object's write() method.

        Example usage:

        >>> import cplex
        >>> with cplex.Cplex() as c, open("output.txt", "w") as f:
        ...     output = c.set_error_stream(f)
        ...     output.write("this is an example")
        """
        return self._env.set_error_stream(error_file, fn)

    def set_log_stream(self, log_file, fn=None):
        """Specifies where the log will be printed.

        The first argument must be a file-like object (i.e., an object
        with a write method and a flush method). Use None as the first
        argument to suppress output.

        The second optional argument is a function that takes a string as
        input and returns a string. If specified, strings sent to this
        stream will be processed by this function before being written.

        Returns the stream to which the log will be written. To write to
        this stream, use this object's write() method.

        >>> import cplex
        >>> with cplex.Cplex() as c, open("output.txt", "w") as f:
        ...     output = c.set_log_stream(f)
        ...     output.write("this is an example")
        """
        return self._env.set_log_stream(log_file, fn)

    def get_version(self):
        """Returns a string specifying the version of CPLEX.

        See :cpxapi:`CPXversion` in the Callable Library Reference Manual
        for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> version = c.get_version()
        """
        return self._env.get_version()

    def get_versionnumber(self):
        """Returns an integer specifying the version of CPLEX.

        The version of CPLEX is in the format vvrrmmff, where vv is the
        version, rr is the release, mm is the modification, and ff is the
        fixpack number. For example, for CPLEX version 12.5.0.1 the
        returned value is 12050001.

        See :cpxapi:`CPXversionnumber` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> versionnumber = c.get_versionnumber()
        """
        return self._env.get_versionnumber()

    def get_num_cores(self):
        """Returns the number of cores on this machine.

        See :cpxapi:`CPXgetnumcores` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> num_cores = c.get_num_cores()
        """
        return self._env.get_num_cores()

    def get_stats(self):
        """Returns a `Stats` object containing problem statistics.

        Note
          Printing the `Stats` object will give a nice summary of the
          problem statistics in human readable form (e.g. as with the
          "display problem statistics" command in the CPLEX interactive).

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> stats = c.get_stats()
        >>> stats.num_variables
        32
        >>> stats.num_linear_constraints
        27
        """
        return Stats(self)

    def get_time(self):
        """Returns a time stamp in seconds.

        To measure time spent between a starting point and ending point
        of an operation, take the result of this method at the starting
        point; take the result of this method at the end point; subtract
        the starting time stamp from the ending time stamp; the
        subtraction yields elapsed time in seconds.

        The interpretation of this value as wall clock time or CPU time
        is controlled by the parameter clocktype.

        The absolute value of the time stamp is not meaningful.

        See :cpxapi:`CPXgettime` in the Callable Library Reference Manual
        for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> start = c.get_time()
        >>> c.solve()
        >>> solve_time = c.get_time() - start
        """
        return self._env.get_time()

    def get_dettime(self):
        """Returns a deterministic time stamp in ticks.

        To measure elapsed deterministic time in ticks between a starting
        point and ending point of an operation, take the deterministic
        time stamp at the starting point; take the deterministic time
        stamp at the ending point; subtract the starting deterministic
        time stamp from the ending deterministic time stamp.

        The absolute value of the deterministic time stamp is not
        meaningful.

        See :cpxapi:`CPXgetdettime` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> out = c.set_results_stream(None)
        >>> out = c.set_log_stream(None)
        >>> c.read("lpex.mps")
        >>> start = c.get_dettime()
        >>> c.solve()
        >>> solve_dettime = c.get_dettime() - start
        """
        return self._env.get_dettime()

    def use_aborter(self, aborter):
        """Use an `Aborter` to control termination of solve methods.

        Instructs the invoking object to use the aborter to control
        termination of its solving and tuning methods.

        If another aborter is already being used by the invoking object,
        then this method overrides the previously used aborter.

        Returns the aborter installed in the invoking object or None.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> aborter = c.use_aborter(cplex.Aborter())
        """
        self.remove_aborter()
        self._aborter = aborter
        if self._aborter is not None:
            _proc.setterminate(self._env._e, self._env_lp_ptr,
                               self._aborter._p)
            self._aborter._register(self)
        return self._aborter

    def remove_aborter(self):
        """Removes the `Aborter` being used by the invoking object.

        Returns the aborter that was removed or None.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> aborter = c.use_aborter(cplex.Aborter())
        >>> aborter = c.remove_aborter()
        """
        aborter, self._aborter = (self._aborter, None)
        _proc.setterminate(self._env._e, self._env_lp_ptr, None)
        if aborter is not None:
            aborter._unregister(self)
        return aborter

    def get_aborter(self):
        """Returns the `Aborter` being used by the invoking object.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> aborter = c.use_aborter(cplex.Aborter())
        >>> aborter = c.get_aborter()
        """
        return self._aborter

    def set_callback(self, functor=None, contextmask=0):
        """Set callback function to use during optimization.

        Sets the callback that CPLEX invokes during optimization. If
        functor is None then contextmask will be treated as 0 and the
        callback is effectively cleared from CPLEX.

        In all other cases functor must be a reference to an object that
        has a callable member called 'invoke' (if that does not exist, or
        is not a callable, an exception will occur the first time CPLEX
        attempts to invoke the callback). Whenever CPLEX needs to invoke
        the callback it calls this member with exactly one argument: an
        instance of `cplex.callbacks.Context`.

        Note that in the 'invoke' function you must not invoke any
        functions of the Cplex instance that is performing the current
        solve. All functions that can be invoked from a callback are
        members of the `cplex.callbacks.Context` class.

        contextmask must be the bitwise OR of values from
        `cplex.callbacks.Context.id` and specifies in which contexts
        CPLEX shall invoke the callback: the callback is invoked in all
        contexts for which the corresponding bit is set in contextmask.

        Note about cplex.callbacks.Context.id.thread_down: This is
        considered a "destructor" function and should not raise any
        exception. Any exception raised from the callback in this context
        will just be ignored.

        See `cplex.callbacks.Context`.

        See :cpxapi:`CPXcallbacksetfunc` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> class GenericCB():
        ...     def invoke(self, context):
        ...         pass  # Do something here.
        >>> cb = GenericCB()
        >>> c.set_callback(cb)  # Register callback.
        >>> c.set_callback(None)  # Clear callback.
        """
        # First of all, clear any existing callback
        self._genericcallback = None
        self._genericcontextmask = None
        _proc.setgenericcallbackfunc(self._env._e, self._lp, contextmask, None)
        # TODO: Use hasattr() or similar to check whether 'functor' has
        #        a method called 'invoke'? This is never a complete
        #        guard since the attribute may be deleted from the instance
        #        later. So for now we just don't do it.
        # FIXME: This is very shaky since the callback will be deleted
        #        whenever we create a new self._lp :-( So far I don't see us
        #        deleting/recreating self._lp anywhere, but if that ever
        #        happens we have to be careful.
        if contextmask != 0 and functor is not None:
            _proc.setgenericcallbackfunc(self._env._e, self._lp, contextmask,
                                         self)
            self._genericcallback = functor
            self._genericcontextmask = contextmask

    def _invoke_generic_callback(self, contextptr, contextid):
        """non-public"""
        # This is invoked by the cpxpygenericcallbackfuncwrap() trampoline
        # function in the native code and is responsible for invoking the
        # user callback.
        context = callbacks.Context(weakref.proxy(self), contextptr, contextid)
        if context.get_id() == callbacks.Context.id.thread_down:
            # For thread_down we ignore any exception
            try:
                self._genericcallback.invoke(context)
            except:  # noqa: E722
                pass
        else:
            self._genericcallback.invoke(context)

    def set_modeling_assistance_callback(self, functor=None):
        """Set callback function to use for modeling assistance warnings.

        Sets the callback that CPLEX invokes before and after
        optimization (once for every modeling issue detected). If functor
        is None then the callback is effectively cleared from CPLEX. The
        callback function will only be invoked if the CPLEX parameter
        Cplex.parameters.read.datacheck is set to
        Cplex.parameters.read.datacheck.values.assist (2). In addition,
        the parameter Cplex.parameters.read.warninglimit controls the
        number of times each type of modeling assistance warning will be
        reported (the rest will be ignored). See CPX_PARAM_DATACHECK and
        CPX_PARAM_WARNLIM in the Parameters of CPLEX Reference Manual.

        In all other cases functor must be a reference to an object that
        has a callable attribute named 'invoke' (if that does not exist,
        or is not a callable, an exception will occur the first time CPLEX
        attempts to invoke the callback). Whenever CPLEX needs to invoke
        the callback it calls this member with two argument: the modeling
        issue ID and the associated warning message.

        See `model_info`.

        See :cpxapi:`CPXmodelasstcallbacksetfunc` in the Callable Library
        Reference Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.parameters.read.datacheck.set(
        ...     c.parameters.read.datacheck.values.assist)
        >>> class ModelAsstCB():
        ...     def invoke(self, issueid, message):
        ...         pass  # Do something here.
        >>> cb = ModelAsstCB()
        >>> c.set_modeling_assistance_callback(cb)  # Register callback.
        >>> c.set_modeling_assistance_callback(None)  # Clear callback.
        """
        # First of all, clear any existing callback
        self._modelasstcb = None
        _proc.modelasstcallbacksetfunc(self._env._e, self._lp, None)
        # We could use hasattr() or similar to check whether 'functor'
        # has a method called 'invoke'. This is never a complete guard
        # since the attribute may be deleted from the instance later. So,
        # for now, we just don't check anything.
        # FIXME: See FIXME in set_callback above.
        if functor is not None:
            _proc.modelasstcallbacksetfunc(self._env._e, self._lp, self)
            self._modelasstcb = functor

    def _invoke_modelasst_callback(self, issueid, message):
        """non-public"""
        # This is invoked by the cpxpymodelasstcallbackfuncwrap()
        # trampoline function in the native code and is responsible for
        # invoking the user callback.
        self._modelasstcb.invoke(issueid, message)

    def create_parameter_set(self):
        """Returns a new CPLEX parameter set object that is associated
        with this CPLEX problem object.

        Note
          When this CPLEX problem object is destroyed, the parameter set
          object returned by this function will also be destoyed.

        See `ParameterSet`.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        >>> len(ps)
        1
        """
        ps = ParameterSet(self._env)
        self._pslst.append(ps)
        return ps

    def copy_parameter_set(self, source):
        """Returns a deep copy of a parameter set.

        In a sense, this a convenience function; it is equivalent to
        querying what parameters are in the source parameter set,
        querying their values, and then adding those parameters to the
        target parameter set.

        Note
          The source parameter set must have been created by this CPLEX
          problem object. Mixing parameter sets from different CPLEX
          problem objects is not supported.

        Note
          When this CPLEX problem object is destroyed, the parameter set
          object returned by this function will also be destoyed.

        See `ParameterSet`.

        See :cpxapi:`CPXparamsetcopy` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> source = c.create_parameter_set()
        >>> source.add(c.parameters.advance,
        ...            c.parameters.advance.values.none)
        >>> len(source)
        1
        >>> target = c.copy_parameter_set(source)
        >>> len(target)
        1
        """
        if not isinstance(source, ParameterSet):
            raise TypeError("source must be a ParameterSet")
        if source not in self._pslst:
            raise ValueError("parameter set must have been created"
                             " by this CPLEX problem object")
        target = ParameterSet(self._env)
        self._pslst.append(target)
        _proc.paramsetcopy(self._env._e, target._ps, source._ps)
        return target

    def get_parameter_set(self):
        """Returns a parameter set containing parameters that have been
        changed from their default values in the environment.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.parameters.advance.set(c.parameters.advance.values.none)
        >>> ps = c.get_parameter_set()
        >>> val = ps.get(c.parameters.advance)
        >>> val == c.parameters.advance.values.none
        True
        """
        ps = ParameterSet(self._env)
        self._pslst.append(ps)
        for param, value in self.parameters.get_changed():
            ps.add(param._id, value)
        return ps

    def set_parameter_set(self, source):
        """Applies the parameter values in the paramset to the
        environment.

        Note
          The source parameter set must have been created by this CPLEX
          problem object. Mixing parameter sets from different CPLEX
          problem objects is not supported.

        See :cpxapi:`CPXparamsetapply` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> ps = c.create_parameter_set()
        >>> ps.add(c.parameters.advance,
        ...        c.parameters.advance.values.none)
        >>> c.set_parameter_set(ps)
        >>> value = c.parameters.advance.get()
        >>> value == c.parameters.advance.values.none
        True
        """
        if not isinstance(source, ParameterSet):
            raise TypeError("source must be a ParameterSet")
        if source not in self._pslst:
            raise ValueError("parameter set must have been created"
                             " by this CPLEX problem object")
        _proc.paramsetapply(self._env._e, source._ps)

    def copylp(self, numcols, numrows, objsense=ObjSense.minimize,
               obj=None, rhs=None, senses="",
               matbeg=None, matcnt=None, matind=None, matval=None,
               lb=None, ub=None, range_values=None, colnames=None,
               rownames=None):
        """Copies LP data into a CPLEX problem object.

        The arguments define an objective function, constraint matrix,
        variable bounds, righthand side, constraint senses, range values,
        names of constraints, and names of variables.

        Note
          This method can give better performance when building a model,
          but it may not be as user friendly as using other methods. To
          compare different techniques, see the lpex1.py example.

        Note
          Calling this method destroys any existing data associated with
          the problem object.

        numcols : the number of columns in the constraint matrix, or
        equivalently, the number of variables in the problem object.

        numrows : the number of rows in the constraint matrix, not
        including the objective function or bounds on the variables.

        objsense : sets the sense of the objective function. Must be
        either Cplex.objective.sense.minimize or
        Cplex.objective.sense.maximize.

        obj : a list of floats of length at least ``numcols`` containing
        the objective function coefficients. Required if ``numcols`` > 0.

        rhs : a list of floats of length at least ``numrows`` containing
        the righthand side value for each constraint in the constraint
        matrix. Required if ``numrows`` > 0.

        senses : A list of single-character strings or a string
        containing the sense of each constraint in the constraint matrix.
        Must be of length at least ``numrows``. Each entry must be one of
        'G', 'L', 'E', and 'R', indicating greater-than-or-equal-to (>=),
        less-than-or-equal-to (<=), equality (=), and ranged constraints,
        respectively. Required if ``numrows`` > 0.

        With respect to the arguments ``matbeg`` (beginning of the matrix),
        ``matcnt`` (count of the matrix), ``matind`` (indices of the matrix),
        and ``matval`` (values of the matrix), CPLEX needs to know only the
        nonzero coefficients. These arguments are required if
        ``numcols`` > 0 and ``numrows`` > 0.

        These arrays are accessed as follows. Suppose that CPLEX wants to
        access the entries in some column j. These are assumed to be
        given by the entries:
        matval[matbeg[j]],.., matval[matbeg[j]+matcnt[j]-1]

        The corresponding row indices are:
        matind[matbeg[j]],.., matind[matbeg[j]+matcnt[j]-1]

        lb : a list of length at least ``numcols`` containing the lower
        bound on each of the variables. Required if ``numcols`` > 0.

        ub : a list of length at least ``numcols`` containing the upper
        bound on each of the variables. Required if ``numcols`` > 0.

        range_values : a list of floats, specifying the difference
        between lefthand side and righthand side of each linear
        constraint. If range_values[i] > 0 (zero) then the constraint i
        is defined as rhs[i] <= rhs[i] + range_values[i]. If
        range_values[i] < 0 (zero) then constraint i is defined as
        rhs[i] + range_value[i] <= a*x <= rhs[i].

        colnames : a list of strings of length at least ``numcols``
        containing the names of the matrix columns or, equivalently, the
        constraint names.

        rownames : a list of strings of length at least ``numrows``
        containing the names of the matrix rows or, equivalently, the
        constraint names.

        See :cpxapi:`CPXcopylpwnames` in the Callable Library Reference
        Manual for more detail.

        Example usage:

        >>> import cplex
        >>> c = cplex.Cplex()
        >>> c.copylp(numcols=3,
        ...          numrows=2,
        ...          objsense=c.objective.sense.maximize,
        ...          obj=[1.0, 2.0, 3.0],
        ...          rhs=[20.0, 30.0],
        ...          senses="LL",
        ...          matbeg=[0, 2, 4],
        ...          matcnt=[2, 2, 2],
        ...          matind=[0, 1, 0, 1, 0, 1],
        ...          matval=[-1.0, 1.0, 1.0, -3.0, 1.0, 1.0],
        ...          lb=[0.0, 0.0, 0.0],
        ...          ub=[40.0, cplex.infinity, cplex.infinity],
        ...          range_values=[0.0, 0.0],
        ...          colnames=["x1", "x2", "x3"],
        ...          rownames=["c1", "c2"])
        """
        (obj, rhs, senses, matbeg,
         matcnt, matind, matval,
         lb, ub, range_values,
         colnames, rownames) = init_list_args(obj, rhs, senses, matbeg, matcnt,
                                              matind, matval, lb, ub,
                                              range_values, colnames, rownames)
        if not isinstance(senses, str):
            senses = "".join(senses)
        # Check arg lengths to avoid potential segfault in the C API.
        validate_arg_lengths([obj, lb, ub, colnames],
                             extra_msg=": obj, lb, ub, colnames")
        # Special case: Check that numcols <= len(obj).
        if numcols > 0 and numcols > len(obj):
            raise CplexError("inconsistent arguments: numcols > len(obj)")
        validate_arg_lengths([rhs, senses, range_values, rownames],
                             extra_msg=": rhs, senses, range_values, rownames")
        # Special case: Check that numrows <= len(rhs).
        if numrows > 0 and numrows > len(rhs):
            raise CplexError("inconsistent arguments: numrows > len(rhs)")
        validate_arg_lengths([matbeg, matcnt], extra_msg=": matbeg, matcnt")
        validate_arg_lengths([matind, matval], extra_msg=": matind, matval")
        # Check special case: that len(matind) == sum(matcnt). We don't
        # have to check matval b/c of validate_arg_lengths check above.
        nnz = sum(matcnt)
        nmatind = len(matind)
        if nmatind > 0 and nmatind != nnz:
            raise CplexError(
                "inconsistent arguments: len(matind) != sum(matcnt)")
        _proc.copylpwnames(self._env._e, self._lp, numcols, numrows,
                           objsense, obj, rhs, senses,
                           matbeg, matcnt, matind, matval,
                           lb, ub, range_values,
                           colnames, rownames)
