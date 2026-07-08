# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Tests get_stats() and the Stats class.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
import cplex._internal._constants as _constants
from cplex.callbacks import UserCutCallback, LazyConstraintCallback
from cplex import SparsePair

CPX_INFBOUND = _constants.CPX_INFBOUND
CASO8_EXAMPLE_FILE = "../../../examples/data/caso8.mps"
QP_EXAMPLE_FILE = "../../../examples/data/qpex.lp"
INDICATOR_EXAMPLE_FILE = "../../../examples/data/indicator.lp"
SOS_EXAMPLE_FILE = "../../../examples/data/sosex3.lp"
QCP_EXAMPLE_FILE = "../../../examples/data/qcp.lp"
NOSWOT_EXAMPLE_FILE = "../../../examples/data/noswot.mps"
PWL_EXAMPLE_FILE = "../../../examples/data/transport.lp"
MULTIOBJ_EXAMPLE_FILE = "../../../examples/data/dietmultiobj.lp"


class MyUserCutCallback(UserCutCallback):
    """Copied from admipex5.py."""

    def __init__(self, env):
        super().__init__(env)
        self.initcuts()
        self.cuts_added = False

    def initcuts(self):
        self.lhs = [SparsePair(ind=["X21", "X22"], val=[1.0, -1.]),
                    SparsePair(ind=["X22", "X23"], val=[1.0, -1.]),
                    SparsePair(ind=["X23", "X24"], val=[1.0, -1.]),
                    SparsePair(ind=["X11", "X21", "X31", "X41", "X51",
                                    "W11", "W21", "W31", "W41", "W51"],
                               val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                    0.25, 0.25, 0.25, 0.25, 0.25]),
                    SparsePair(ind=["X12", "X22", "X32", "X42", "X52",
                                    "W12", "W22", "W32", "W42", "W52"],
                               val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                    0.25, 0.25, 0.25, 0.25, 0.25]),
                    SparsePair(ind=["X13", "X23", "X33", "X43", "X53",
                                    "W13", "W23", "W33", "W43", "W53"],
                               val=[2.08, 2.98, 3.4722, 2.24, 2.08,
                                    0.25, + 0.25, 0.25, 0.25, 0.25]),
                    SparsePair(ind=["X14", "X24", "X34", "X44", "X54",
                                    "W14", "W24", "W34", "W44", "W54"],
                               val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                    0.25, 0.25, 0.25, 0.25, 0.25]),
                    SparsePair(ind=["X15", "X25", "X35", "X45", "X55",
                                    "W15", "W25", "W35", "W45", "W55"],
                               val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                    0.25, 0.25, 0.25, 0.25, 0.25])]
        self.rhs = [0.0, 0.0, 0.0, 20.25, 20.25, 20.25, 20.25, 16.25]

    def __call__(self):
        # loop through our list of cuts and check whether they are violated
        ncuts = len(self.rhs)
        for i in range(ncuts):
            # calculate activity of cut
            act = 0
            cutlen = len(self.lhs[i].ind)
            for k in range(cutlen):
                j = self.lhs[i].ind[k]
                a = self.lhs[i].val[k]
                act += a * self.get_values(j)
            # check if cut is violated
            if act > self.rhs[i] + 1e-6:
                self.add(cut=self.lhs[i], sense ="L", rhs=self.rhs[i])
                self.cuts_added = True


class MyLazyConstraintCallback(LazyConstraintCallback):
    """Copied from admipex5.py."""

    def __init__(self, env):
        super().__init__(env)
        self.cuts_added = False

    def __call__(self):
        indices = ["W11", "W12", "W13", "W14", "W15"]
        act = 0.0
        for i in indices:
            act += self.get_values(i)
        if act > 3.01:
            self.add(constraint=SparsePair(ind=indices, val=[1.0] * 5),
                     sense="L", rhs=3.0)
            self.cuts_added = True


class GetStatsTestCase(CplexTestCase):
    """Base class for get_stats() tests."""

    def checkAll(self, stats):
        self.assertFalse(stats is None)
        self.checkName(stats)
        self.checkVariableData(stats)
        self.checkLinearConstraintData(stats)
        self.checkIndicatorData(stats)
        self.checkQuadraticConstraintData(stats)
        self.checkSosData(stats)
        self.checkLazyConstraintData(stats)
        self.checkUserCutData(stats)
        self.checkPwlData(stats)
        self.checkMinMaxVariableData(stats)
        self.checkMinMaxLinearConstraintData(stats)
        self.checkMinMaxQuadraticConstraintData(stats)
        self.checkMinMaxIndicatorConstraintData(stats)
        self.checkMinMaxLazyConstraintData(stats)
        self.checkMinMaxUserCutData(stats)
        self.checkString(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, '')

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 0)
        self.assertEqual(stats.num_nonnegative, 0)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 0)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 0)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 0)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 0)
        self.assertEqual(stats.num_linear_less, 0)
        self.assertEqual(stats.num_linear_equal, 0)
        self.assertEqual(stats.num_linear_greater, 0)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 0)
        self.assertEqual(stats.num_linear_rhs_nz, 0)

    def checkIndicatorData(self, stats):
        self.assertEqual(stats.num_indicator_constraints, 0)
        self.assertEqual(stats.num_indicator_less, 0)
        self.assertEqual(stats.num_indicator_equal, 0)
        self.assertEqual(stats.num_indicator_greater, 0)
        self.assertEqual(stats.num_indicator_complemented, 0)
        self.assertEqual(stats.num_indicator_nz, 0)
        self.assertEqual(stats.num_indicator_rhs_nz, 0)

    def checkQuadraticConstraintData(self, stats):
        self.assertEqual(stats.num_quadratic_constraints, 0)
        self.assertEqual(stats.num_quadratic_less, 0)
        self.assertEqual(stats.num_quadratic_greater, 0)
        self.assertEqual(stats.num_quadratic_linear_nz, 0)
        self.assertEqual(stats.num_quadratic_nz, 0)
        self.assertEqual(stats.num_quadratic_rhs_nz, 0)

    def checkSosData(self, stats):
        self.assertEqual(stats.num_SOS_constraints, 0)
        self.assertEqual(stats.num_SOS1, 0)
        self.assertEqual(stats.num_SOS1_members, 0)
        self.assertEqual(stats.type_SOS1, '')
        self.assertEqual(stats.num_SOS2, 0)
        self.assertEqual(stats.num_SOS2_members, 0)
        self.assertEqual(stats.type_SOS2, '')

    def checkLazyConstraintData(self, stats):
        self.assertEqual(stats.num_lazy_constraints, 0)
        self.assertEqual(stats.num_lazy_nnz, 0)
        self.assertEqual(stats.num_lazy_lt, 0)
        self.assertEqual(stats.num_lazy_eq, 0)
        self.assertEqual(stats.num_lazy_gt, 0)
        self.assertEqual(stats.num_lazy_rhs_nnz, 0)

    def checkUserCutData(self, stats):
        self.assertEqual(stats.num_user_cuts, 0)
        self.assertEqual(stats.num_user_cuts_nnz, 0)
        self.assertEqual(stats.num_user_cuts_lt, 0)
        self.assertEqual(stats.num_user_cuts_eq, 0)
        self.assertEqual(stats.num_user_cuts_gt, 0)
        self.assertEqual(stats.num_user_cuts_rhs_nnz, 0)

    def checkPwlData(self, stats):
        self.assertEqual(stats.num_pwl_constraints, 0)
        self.assertEqual(stats.num_pwl_breaks, 0)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, -CPX_INFBOUND)
        self.assertEqual(stats.max_upper_bound, CPX_INFBOUND)
        self.assertEqual(stats.min_linear_objective, -CPX_INFBOUND)
        self.assertEqual(stats.max_linear_objective, CPX_INFBOUND)
        if stats.num_quadratic_objective_nz > 0:
            self.assertEqual(stats.min_quadratic_objective, 0)
            self.assertEqual(stats.max_quadratic_objective, 0)
        else:
            self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
            self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, -CPX_INFBOUND)
        self.assertEqual(stats.max_linear_constraints, CPX_INFBOUND)
        self.assertEqual(stats.min_linear_constraints_rhs, -CPX_INFBOUND)
        self.assertEqual(stats.max_linear_constraints_rhs, CPX_INFBOUND)
        if stats.num_linear_range > 0:
            self.assertEqual(stats.min_linear_range, 0)
            self.assertEqual(stats.max_linear_range, 0)
        else:
            self.assertFalse(hasattr(stats, 'min_linear_range'))
            self.assertFalse(hasattr(stats, 'max_linear_range'))

    def checkMinMaxQuadraticConstraintData(self, stats):
        if stats.num_quadratic_constraints > 0:
            self.assertEqual(stats.min_quadratic_linear, 0)
            self.assertEqual(stats.max_quadratic_linear, 0)
            self.assertEqual(stats.min_quadratic, 0)
            self.assertEqual(stats.max_quadratic, 0)
            self.assertEqual(stats.min_quadratic_rhs, 0)
            self.assertEqual(stats.max_quadratic_rhs, 0)
        else:
            self.assertFalse(hasattr(stats, 'min_quadratic_linear'))
            self.assertFalse(hasattr(stats, 'max_quadratic_linear'))
            self.assertFalse(hasattr(stats, 'min_quadratic'))
            self.assertFalse(hasattr(stats, 'max_quadratic'))
            self.assertFalse(hasattr(stats, 'min_quadratic_rhs'))
            self.assertFalse(hasattr(stats, 'max_quadratic_rhs'))

    def checkMinMaxIndicatorConstraintData(self, stats):
        if stats.num_indicator_constraints > 0:
            self.assertEqual(stats.min_indicator, 0)
            self.assertEqual(stats.max_indicator, 0)
            self.assertEqual(stats.min_indicator_rhs, 0)
            self.assertEqual(stats.max_indicator_rhs, 0)
        else:
            self.assertFalse(hasattr(stats, 'min_indicator'))
            self.assertFalse(hasattr(stats, 'max_indicator'))
            self.assertFalse(hasattr(stats, 'min_indicator_rhs'))
            self.assertFalse(hasattr(stats, 'max_indicator_rhs'))

    def checkMinMaxLazyConstraintData(self, stats):
        if stats.num_lazy_constraints > 0:
            self.assertEqual(stats.min_lazy_constraint, 0)
            self.assertEqual(stats.max_lazy_constraint, 0)
            self.assertEqual(stats.min_lazy_constraint_rhs, 0)
            self.assertEqual(stats.max_lazy_constraint_rhs, 0)
        else:
            self.assertFalse(hasattr(stats, 'min_lazy_constraint'))
            self.assertFalse(hasattr(stats, 'max_lazy_constraint'))
            self.assertFalse(hasattr(stats, 'min_lazy_constraint_rhs'))
            self.assertFalse(hasattr(stats, 'max_lazy_constraint_rhs'))

    def checkMinMaxUserCutData(self, stats):
        if stats.num_user_cuts > 0:
            self.assertEqual(stats.min_user_cut, 0)
            self.assertEqual(stats.max_user_cut, 0)
            self.assertEqual(stats.min_user_cut_rhs, 0)
            self.assertEqual(stats.max_user_cut_rhs, 0)
        else:
            self.assertFalse(hasattr(stats, 'min_user_cut'))
            self.assertFalse(hasattr(stats, 'max_user_cut'))
            self.assertFalse(hasattr(stats, 'min_user_cut_rhs'))
            self.assertFalse(hasattr(stats, 'max_user_cut_rhs'))

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : 
Objective sense      : Minimize
Variables            :       0
Objective nonzeros   :       0
Linear constraints   :       0
  Nonzeros           :       0
  RHS nonzeros       :       0

Variables            : Min LB: all infinite     Max UB: all infinite   
Objective nonzeros   : Min   : all zero         Max   : all zero       
Linear constraints   :
  Nonzeros           : Min   : all zero         Max   : all zero       
  RHS nonzeros       : Min   : all zero         Max   : all zero       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsEmptyTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        stats = cpx.get_stats()
        self.checkAll(stats)


class GetStatsCaso8Test(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.read(CASO8_EXAMPLE_FILE)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, CASO8_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 1282)
        self.assertEqual(stats.num_nonnegative, 0)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 5)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 1277)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 5)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 236)
        self.assertEqual(stats.num_linear_less, 136)
        self.assertEqual(stats.num_linear_equal, 0)
        self.assertEqual(stats.num_linear_greater, 100)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 40864)
        self.assertEqual(stats.num_linear_rhs_nz, 236)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, 10000000000.0)
        self.assertEqual(stats.min_linear_objective, 0.001)
        self.assertEqual(stats.max_linear_objective, 1.0)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 7.4609e-05)
        self.assertEqual(stats.max_linear_constraints, 165650.0)
        self.assertEqual(stats.min_linear_constraints_rhs, 1.0)
        self.assertEqual(stats.max_linear_constraints_rhs, 7643.5)
        self.assertFalse(stats.num_linear_range > 0)
        self.assertFalse(hasattr(stats, 'min_linear_range'))
        self.assertFalse(hasattr(stats, 'max_linear_range'))

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : ../../../examples/data/caso8.mps
Objective sense      : Maximize
Variables            :    1282  [Box: 5,  Binary: 1277]
Objective nonzeros   :       5
Linear constraints   :     236  [Less: 136,  Greater: 100]
  Nonzeros           :   40864
  RHS nonzeros       :     236

Variables            : Min LB: 0.000000         Max UB: 1.000000e+10   
Objective nonzeros   : Min   : 0.001000000      Max   : 1.000000       
Linear constraints   :
  Nonzeros           : Min   : 7.460900e-05     Max   : 165650.0       
  RHS nonzeros       : Min   : 1.000000         Max   : 7643.500       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsQpTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.read(QP_EXAMPLE_FILE)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, QP_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 3)
        self.assertEqual(stats.num_nonnegative, 2)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 1)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 0)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 3)
        self.assertEqual(stats.num_linear_objective_nz, 3)
        self.assertEqual(stats.num_quadratic_objective_nz, 7)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 2)
        self.assertEqual(stats.num_linear_less, 2)
        self.assertEqual(stats.num_linear_equal, 0)
        self.assertEqual(stats.num_linear_greater, 0)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 6)
        self.assertEqual(stats.num_linear_rhs_nz, 2)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, 40)
        self.assertEqual(stats.min_linear_objective, 1)
        self.assertEqual(stats.max_linear_objective, 3)
        self.assertTrue(stats.num_quadratic_objective_nz > 0)
        self.assertEqual(stats.min_quadratic_objective, 6)
        self.assertEqual(stats.max_quadratic_objective, 33)

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 1)
        self.assertEqual(stats.max_linear_constraints, 3)
        self.assertEqual(stats.min_linear_constraints_rhs, 20)
        self.assertEqual(stats.max_linear_constraints_rhs, 30)
        self.assertFalse(stats.num_linear_range > 0)
        self.assertFalse(hasattr(stats, 'min_linear_range'))
        self.assertFalse(hasattr(stats, 'max_linear_range'))

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : ../../../examples/data/qpex.lp
Objective sense      : Maximize
Variables            :       3  [Nneg: 2,  Box: 1,  Qobj: 3]
Objective nonzeros   :       3
Objective Q nonzeros :       7
Linear constraints   :       2  [Less: 2]
  Nonzeros           :       6
  RHS nonzeros       :       2

Variables            : Min LB: 0.000000         Max UB: 40.00000       
Objective nonzeros   : Min   : 1.000000         Max   : 3.000000       
Objective Q nonzeros : Min   : 6.000000         Max   : 33.00000       
Linear constraints   :
  Nonzeros           : Min   : 1.000000         Max   : 3.000000       
  RHS nonzeros       : Min   : 20.00000         Max   : 30.00000       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsIndicatorTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.read(INDICATOR_EXAMPLE_FILE)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, INDICATOR_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 2)
        self.assertEqual(stats.num_nonnegative, 0)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 1)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 1)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 2)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkIndicatorData(self, stats):
        self.assertEqual(stats.num_indicator_constraints, 18)
        self.assertEqual(stats.num_indicator_less, 6)
        self.assertEqual(stats.num_indicator_equal, 6)
        self.assertEqual(stats.num_indicator_greater, 6)
        self.assertEqual(stats.num_indicator_complemented, 9)
        self.assertEqual(stats.num_indicator_nz, 18)
        self.assertEqual(stats.num_indicator_rhs_nz, 18)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, 10)
        self.assertEqual(stats.min_linear_objective, 1)
        self.assertEqual(stats.max_linear_objective, 5)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxIndicatorConstraintData(self, stats):
        self.assertTrue(stats.num_indicator_constraints > 0)
        self.assertEqual(stats.min_indicator, 10)
        self.assertEqual(stats.max_indicator, 10)
        self.assertEqual(stats.min_indicator_rhs, 1)
        self.assertEqual(stats.max_indicator_rhs, 1)

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : ../../../examples/data/indicator.lp
Objective sense      : Maximize
Variables            :       2  [Box: 1,  Binary: 1]
Objective nonzeros   :       2
Linear constraints   :       0
  Nonzeros           :       0
  RHS nonzeros       :       0
Indicator constraints:      18  [Less: 6,  Equal: 6,  Greater: 6]
  Complemented       :       9
  Nonzeros           :      18
  RHS nonzeros       :      18

Variables            : Min LB: 0.000000         Max UB: 10.00000       
Objective nonzeros   : Min   : 1.000000         Max   : 5.000000       
Linear constraints   :
  Nonzeros           : Min   : all zero         Max   : all zero       
  RHS nonzeros       : Min   : all zero         Max   : all zero       
Indicator constraints:
  Nonzeros           : Min   : 10.00000         Max   : 10.00000       
  RHS nonzeros       : Min   : 1.000000         Max   : 1.000000       
"""
        self.assertEqual(stats_str, expected_stats_str)

class GetStatsSosTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.read(SOS_EXAMPLE_FILE)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, SOS_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 33)
        self.assertEqual(stats.num_nonnegative, 0)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 14)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 19)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 33)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 20)
        self.assertEqual(stats.num_linear_less, 16)
        self.assertEqual(stats.num_linear_equal, 0)
        self.assertEqual(stats.num_linear_greater, 4)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 112)
        self.assertEqual(stats.num_linear_rhs_nz, 19)

    def checkSosData(self, stats):
        self.assertEqual(stats.num_SOS_constraints, 4)
        self.assertEqual(stats.num_SOS1, 4)
        self.assertEqual(stats.num_SOS1_members, 14)
        self.assertEqual(stats.type_SOS1, 'all continuous')
        self.assertEqual(stats.num_SOS2, 0)
        self.assertEqual(stats.num_SOS2_members, 14)
        self.assertEqual(stats.type_SOS2, 'all continuous')

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, 10)
        self.assertEqual(stats.min_linear_objective, 16.3)
        self.assertEqual(stats.max_linear_objective, 517)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 0.1)
        self.assertEqual(stats.max_linear_constraints, 400)
        self.assertEqual(stats.min_linear_constraints_rhs, 1)
        self.assertEqual(stats.max_linear_constraints_rhs, 2700)
        self.assertFalse(stats.num_linear_range > 0)
        self.assertFalse(hasattr(stats, 'min_linear_range'))
        self.assertFalse(hasattr(stats, 'max_linear_range'))

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : ../../../examples/data/sosex3.lp
Objective sense      : Minimize
Variables            :      33  [Box: 14,  Binary: 19]
Objective nonzeros   :      33
Linear constraints   :      20  [Less: 16,  Greater: 4]
  Nonzeros           :     112
  RHS nonzeros       :      19
SOS                  :       4  [SOS1: 4, 14 members, all continuous]

Variables            : Min LB: 0.000000         Max UB: 10.00000       
Objective nonzeros   : Min   : 16.30000         Max   : 517.0000       
Linear constraints   :
  Nonzeros           : Min   : 0.1000000        Max   : 400.0000       
  RHS nonzeros       : Min   : 1.000000         Max   : 2700.000       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsPwlTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.read(PWL_EXAMPLE_FILE)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, PWL_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 24)
        self.assertEqual(stats.num_nonnegative, 24)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 0)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 0)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 12)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkPwlData(self, stats):
        self.assertEqual(stats.num_pwl_constraints, 12)
        self.assertEqual(stats.num_pwl_breaks, 24)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 7)
        self.assertEqual(stats.num_linear_less, 0)
        self.assertEqual(stats.num_linear_equal, 7)
        self.assertEqual(stats.num_linear_greater, 0)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 24)
        self.assertEqual(stats.num_linear_rhs_nz, 7)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, CPX_INFBOUND)
        self.assertEqual(stats.min_linear_objective, 1.0)
        self.assertEqual(stats.max_linear_objective, 1.0)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 1.0)
        self.assertEqual(stats.max_linear_constraints, 1.0)
        self.assertEqual(stats.min_linear_constraints_rhs, 400.0)
        self.assertEqual(stats.max_linear_constraints_rhs, 1250.0)
        self.assertFalse(stats.num_linear_range > 0)
        self.assertFalse(hasattr(stats, 'min_linear_range'))
        self.assertFalse(hasattr(stats, 'max_linear_range'))

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : ../../../examples/data/transport.lp
Objective sense      : Minimize
Variables            :      24
Objective nonzeros   :      12
Linear constraints   :       7  [Equal: 7]
  Nonzeros           :      24
  RHS nonzeros       :       7
PWL                  :      12  [Breaks: 24]

Variables            : Min LB: 0.000000         Max UB: all infinite   
Objective nonzeros   : Min   : 1.000000         Max   : 1.000000       
Linear constraints   :
  Nonzeros           : Min   : 1.000000         Max   : 1.000000       
  RHS nonzeros       : Min   : 400.0000         Max   : 1250.000       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsQcpTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.read(QCP_EXAMPLE_FILE)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, QCP_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 3)
        self.assertEqual(stats.num_nonnegative, 2)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 1)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 0)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 3)
        self.assertEqual(stats.num_linear_objective_nz, 3)
        self.assertEqual(stats.num_quadratic_objective_nz, 7)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 2)
        self.assertEqual(stats.num_linear_less, 2)
        self.assertEqual(stats.num_linear_equal, 0)
        self.assertEqual(stats.num_linear_greater, 0)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 6)
        self.assertEqual(stats.num_linear_rhs_nz, 2)

    def checkQuadraticConstraintData(self, stats):
        self.assertEqual(stats.num_quadratic_constraints, 1)
        self.assertEqual(stats.num_quadratic_less, 1)
        self.assertEqual(stats.num_quadratic_greater, 0)
        self.assertEqual(stats.num_quadratic_linear_nz, 2)
        self.assertEqual(stats.num_quadratic_nz, 4)
        self.assertEqual(stats.num_quadratic_rhs_nz, 1)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, 40)
        self.assertEqual(stats.min_linear_objective, 1)
        self.assertEqual(stats.max_linear_objective, 3)
        self.assertTrue(stats.num_quadratic_objective_nz > 0)
        self.assertEqual(stats.min_quadratic_objective, 6)
        self.assertEqual(stats.max_quadratic_objective, 33)

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 1)
        self.assertEqual(stats.max_linear_constraints, 3)
        self.assertEqual(stats.min_linear_constraints_rhs, 20)
        self.assertEqual(stats.max_linear_constraints_rhs, 30)
        self.assertFalse(stats.num_linear_range > 0)
        self.assertFalse(hasattr(stats, 'min_linear_range'))
        self.assertFalse(hasattr(stats, 'max_linear_range'))

    def checkMinMaxQuadraticConstraintData(self, stats):
        self.assertTrue(stats.num_quadratic_constraints > 0)
        self.assertEqual(stats.min_quadratic_linear, 1)
        self.assertEqual(stats.max_quadratic_linear, 3)
        self.assertEqual(stats.min_quadratic, 1)
        self.assertEqual(stats.max_quadratic, 2)
        self.assertEqual(stats.min_quadratic_rhs, 2)
        self.assertEqual(stats.max_quadratic_rhs, 2)

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : ../../../examples/data/qcp.lp
Objective sense      : Maximize
Variables            :       3  [Nneg: 2,  Box: 1,  Qobj: 3]
Objective nonzeros   :       3
Objective Q nonzeros :       7
Linear constraints   :       2  [Less: 2]
  Nonzeros           :       6
  RHS nonzeros       :       2
Quadratic constraints:       1  [Less: 1]
  Linear terms       :       2
  Quadratic terms    :       4
  RHS nonzeros       :       1

Variables            : Min LB: 0.000000         Max UB: 40.00000       
Objective nonzeros   : Min   : 1.000000         Max   : 3.000000       
Objective Q nonzeros : Min   : 6.000000         Max   : 33.00000       
Linear constraints   :
  Nonzeros           : Min   : 1.000000         Max   : 3.000000       
  RHS nonzeros       : Min   : 20.00000         Max   : 30.00000       
Quadratic constraints:
  Linear terms       : Min   : 1.000000         Max   : 3.000000       
  Quadratic terms    : Min   : 1.000000         Max   : 2.000000       
  RHS nonzeros       : Min   : 2.000000         Max   : 2.000000       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsNoswotTestCase(GetStatsTestCase):
    """Base class for GetStatsNoswot tests."""

    def checkName(self, stats):
        self.assertEqual(stats.name, NOSWOT_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 128)
        self.assertEqual(stats.num_nonnegative, 28)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 0)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 75)
        self.assertEqual(stats.num_integer, 25)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 25)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 182)
        self.assertEqual(stats.num_linear_less, 54)
        self.assertEqual(stats.num_linear_equal, 2)
        self.assertEqual(stats.num_linear_greater, 126)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 735)
        self.assertEqual(stats.num_linear_rhs_nz, 57)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, 100000)
        self.assertEqual(stats.min_linear_objective, 1)
        self.assertEqual(stats.max_linear_objective, 1)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 0.25)
        self.assertEqual(stats.max_linear_constraints, 21)
        self.assertEqual(stats.min_linear_constraints_rhs, 1)
        self.assertEqual(stats.max_linear_constraints_rhs, 43)
        self.assertFalse(stats.num_linear_range > 0)
        self.assertFalse(hasattr(stats, 'min_linear_range'))
        self.assertFalse(hasattr(stats, 'max_linear_range'))


class GetStatsNoswotTestWithCallbacks(GetStatsNoswotTestCase):
    """Test user cuts and lazy constraints added via callbacks.

    We solve the noswot problem with both user cuts and lazy constraints
    (added via callbacks), as in admipex5.py.  In this case, we expect
    that get_stats will _not_ report on these.
    """

    def testGetStats(self):
        # Solve logic copied from admipex5.py
        cpx = self._newCplex()
        cpx.read(NOSWOT_EXAMPLE_FILE)
        # need to use traditional branch-and-cut to allow for control callbacks
        cpx.parameters.mip.strategy.search.set(
            cpx.parameters.mip.strategy.search.values.traditional)
        # set node log interval to 1000
        cpx.parameters.mip.interval.set(1000)
        # install the user cut callback
        ucb = cpx.register_callback(MyUserCutCallback)
        self.assertFalse(ucb.cuts_added)
        # install the lazy constraint callback
        lcb = cpx.register_callback(MyLazyConstraintCallback)
        self.assertFalse(lcb.cuts_added)
        # solve problem
        cpx.solve()
        self.assertTrue(ucb.cuts_added)
        self.assertTrue(lcb.cuts_added)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : examples/data/noswot.mps
Objective sense      : Minimize
Variables            :     128  [Nneg: 28,  Binary: 75,  General Integer: 25]
Objective nonzeros   :      25
Linear constraints   :     182  [Less: 54,  Greater: 126,  Equal: 2]
  Nonzeros           :     735
  RHS nonzeros       :      57

Variables            : Min LB: 0.000000         Max UB: 100000.0       
Objective nonzeros   : Min   : 1.000000         Max   : 1.000000       
Linear constraints   :
  Nonzeros           : Min   : 0.2500000        Max   : 21.00000       
  RHS nonzeros       : Min   : 1.000000         Max   : 43.00000       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsNoswotTestWithoutCallbacks(GetStatsNoswotTestCase):
    """Test user cuts and lazy constraints in the model.

    We set up the noswot problem with both user cuts and lazy
    constraints, as in admipex5.py, except that we add them to the model
    itself.  In this case, we expect that get_stats _will_ report on
    these.
    """

    def testGetStats(self):
        # Solve logic copied from admipex5.py
        cpx = self._newCplex()
        cpx.read(NOSWOT_EXAMPLE_FILE)
        # need to use traditional branch-and-cut to allow for control callbacks
        cpx.parameters.mip.strategy.search.set(
            cpx.parameters.mip.strategy.search.values.traditional)
        # set node log interval to 1000
        cpx.parameters.mip.interval.set(1000)
        # add the user cuts
        uclhs = [SparsePair(ind=["X21", "X22"], val=[1.0, -1.]),
                 SparsePair(ind=["X22", "X23"], val=[1.0, -1.]),
                 SparsePair(ind=["X23", "X24"], val=[1.0, -1.]),
                 SparsePair(ind=["X11", "X21", "X31", "X41", "X51",
                                 "W11", "W21", "W31", "W41", "W51"],
                            val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                 0.25, 0.25, 0.25, 0.25, 0.25]),
                 SparsePair(ind=["X12", "X22", "X32", "X42", "X52",
                                 "W12", "W22", "W32", "W42", "W52"],
                            val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                 0.25, 0.25, 0.25, 0.25, 0.25]),
                 SparsePair(ind=["X13", "X23", "X33", "X43", "X53",
                                 "W13", "W23", "W33", "W43", "W53"],
                            val=[2.08, 2.98, 3.4722, 2.24, 2.08,
                                 0.25, + 0.25, 0.25, 0.25, 0.25]),
                 SparsePair(ind=["X14", "X24", "X34", "X44", "X54",
                                 "W14", "W24", "W34", "W44", "W54"],
                            val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                 0.25, 0.25, 0.25, 0.25, 0.25]),
                 SparsePair(ind=["X15", "X25", "X35", "X45", "X55",
                                 "W15", "W25", "W35", "W45", "W55"],
                            val=[2.08, 2.98, 3.47, 2.24, 2.08,
                                 0.25, 0.25, 0.25, 0.25, 0.25])]
        ucrhs = [0.0, 0.0, 0.0, 20.25, 20.25, 20.25, 20.25, 16.25]
        cpx.linear_constraints.advanced.add_user_cuts(
            lin_expr=uclhs,
            senses="L" * len(uclhs),
            rhs=ucrhs,
            names=['usercut%s' % s for s in range(len(uclhs))])
        # add the lazy constraint
        cpx.linear_constraints.advanced.add_lazy_constraints(
            lin_expr=[SparsePair(ind=["W11", "W12", "W13", "W14", "W15"],
                                 val=[1.0] * 5)],
            senses="L", rhs=[3.0], names=['lazycut1'])
        # check stats
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkLazyConstraintData(self, stats):
        self.assertEqual(stats.num_lazy_constraints, 1)
        self.assertEqual(stats.num_lazy_nnz, 5)
        self.assertEqual(stats.num_lazy_lt, 1)
        self.assertEqual(stats.num_lazy_eq, 0)
        self.assertEqual(stats.num_lazy_gt, 0)
        self.assertEqual(stats.num_lazy_rhs_nnz, 1)

    def checkUserCutData(self, stats):
        self.assertEqual(stats.num_user_cuts, 8)
        self.assertEqual(stats.num_user_cuts_nnz, 56)
        self.assertEqual(stats.num_user_cuts_lt, 8)
        self.assertEqual(stats.num_user_cuts_eq, 0)
        self.assertEqual(stats.num_user_cuts_gt, 0)
        self.assertEqual(stats.num_user_cuts_rhs_nnz, 5)

    def checkMinMaxLazyConstraintData(self, stats):
        self.assertTrue(stats.num_lazy_constraints > 0)
        self.assertEqual(stats.min_lazy_constraint, 1)
        self.assertEqual(stats.max_lazy_constraint, 1)
        self.assertEqual(stats.min_lazy_constraint_rhs, 3)
        self.assertEqual(stats.max_lazy_constraint_rhs, 3)

    def checkMinMaxUserCutData(self, stats):
        self.assertTrue(stats.num_user_cuts > 0)
        self.assertEqual(stats.min_user_cut, 0.25)
        self.assertEqual(stats.max_user_cut, 3.4722)
        self.assertEqual(stats.min_user_cut_rhs, 16.25)
        self.assertEqual(stats.max_user_cut_rhs, 20.25)

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : examples/data/noswot.mps
Objective sense      : Minimize
Variables            :     128  [Nneg: 28,  Binary: 75,  General Integer: 25]
Objective nonzeros   :      25
Linear constraints   :     182  [Less: 54,  Greater: 126,  Equal: 2]
  Nonzeros           :     735
  RHS nonzeros       :      57

Variables            : Min LB: 0.000000         Max UB: 100000.0       
Objective nonzeros   : Min   : 1.000000         Max   : 1.000000       
Linear constraints   :
  Nonzeros           : Min   : 0.2500000        Max   : 21.00000       
  RHS nonzeros       : Min   : 1.000000         Max   : 43.00000       
Lazy constraints     :
  Nonzeros           : Min   : 1.000000         Max   : 1.000000       
  RHS nonzeros       : Min   : 3.000000         Max   : 3.000000       
User cuts            :
  Nonzeros           : Min   : 0.2500000        Max   : 3.472200       
  RHS nonzeros       : Min   : 16.25000         Max   : 20.25000       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsRangeTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.variables.add(names=["x1", "x2", "x3"])
        cpx.linear_constraints.add(
            lin_expr=[SparsePair(ind=[0, 2], val=[1.0, -1.0]),
                      SparsePair(ind=[0, 1], val=[1.0, 1.0]),
                      SparsePair(ind=[0, 1, 2], val=[-1.0] * 3),
                      SparsePair(ind=[1, 2], val=[10.0, -2.0])],
            senses=["E", "L", "G", "R"],
            rhs=[0.0, 1.0, -1.0, 2.0],
            range_values=[0.0, 0.0, 0.0, -10.0],
            names=["c0", "c1", "c2", "c3"])
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 3)
        self.assertEqual(stats.num_nonnegative, 3)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 0)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 0)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 0)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 4)
        self.assertEqual(stats.num_linear_less, 1)
        self.assertEqual(stats.num_linear_equal, 1)
        self.assertEqual(stats.num_linear_greater, 1)
        self.assertEqual(stats.num_linear_range, 1)
        self.assertEqual(stats.num_linear_nz, 9)
        self.assertEqual(stats.num_linear_rhs_nz, 3)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, CPX_INFBOUND)
        self.assertEqual(stats.min_linear_objective, -CPX_INFBOUND)
        self.assertEqual(stats.max_linear_objective, CPX_INFBOUND)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 1)
        self.assertEqual(stats.max_linear_constraints, 10)
        self.assertEqual(stats.min_linear_constraints_rhs, 1)
        self.assertEqual(stats.max_linear_constraints_rhs, 2)
        self.assertTrue(stats.num_linear_range > 0)
        self.assertEqual(stats.min_linear_range, 10)
        self.assertEqual(stats.max_linear_range, 10)

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : 
Objective sense      : Minimize
Variables            :       3
Objective nonzeros   :       0
Linear constraints   :       4  [Less: 1,  Greater: 1,  Equal: 1,  Range: 1]
  Nonzeros           :       9
  RHS nonzeros       :       3

Variables            : Min LB: 0.000000         Max UB: all infinite   
Objective nonzeros   : Min   : all zero         Max   : all zero       
Linear constraints   :
  Nonzeros           : Min   : 1.000000         Max   : 10.00000       
  RHS nonzeros       : Min   : 1.000000         Max   : 2.000000       
  Range values       : Min   : 10.00000         Max   : 10.00000       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsFixedTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.variables.add(lb=[1], ub=[1])
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 1)
        self.assertEqual(stats.num_nonnegative, 0)
        self.assertEqual(stats.num_fixed, 1)
        self.assertEqual(stats.num_boxed, 0)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 0)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 0)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 1)
        self.assertEqual(stats.max_upper_bound, 1)
        self.assertEqual(stats.min_linear_objective, -CPX_INFBOUND)
        self.assertEqual(stats.max_linear_objective, CPX_INFBOUND)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : 
Objective sense      : Minimize
Variables            :       1  [Fix: 1]
Objective nonzeros   :       0
Linear constraints   :       0
  Nonzeros           :       0
  RHS nonzeros       :       0

Variables            : Min LB: 1.000000         Max UB: 1.000000       
Objective nonzeros   : Min   : all zero         Max   : all zero       
Linear constraints   :
  Nonzeros           : Min   : all zero         Max   : all zero       
  RHS nonzeros       : Min   : all zero         Max   : all zero       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsFreeTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.variables.add(lb=[-cplex.infinity], ub=[cplex.infinity])
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 1)
        self.assertEqual(stats.num_nonnegative, 0)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 0)
        self.assertEqual(stats.num_free, 1)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 0)
        self.assertEqual(stats.num_integer, 0)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 0)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : 
Objective sense      : Minimize
Variables            :       1  [Free: 1]
Objective nonzeros   :       0
Linear constraints   :       0
  Nonzeros           :       0
  RHS nonzeros       :       0

Variables            : Min LB: all infinite     Max UB: all infinite   
Objective nonzeros   : Min   : all zero         Max   : all zero       
Linear constraints   :
  Nonzeros           : Min   : all zero         Max   : all zero       
  RHS nonzeros       : Min   : all zero         Max   : all zero       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsVarTypeTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.variables.add(types=[cpx.variables.type.continuous,
                                 cpx.variables.type.binary,
                                 cpx.variables.type.integer,
                                 cpx.variables.type.semi_integer,
                                 cpx.variables.type.semi_continuous])
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 5)
        self.assertEqual(stats.num_nonnegative, 1)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 0)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 1)
        self.assertEqual(stats.num_integer, 1)
        self.assertEqual(stats.num_semicontinuous, 1)
        self.assertEqual(stats.num_semiinteger, 1)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 0)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0)
        self.assertEqual(stats.max_upper_bound, 1)
        self.assertEqual(stats.min_linear_objective, -CPX_INFBOUND)
        self.assertEqual(stats.max_linear_objective, CPX_INFBOUND)
        self.assertFalse(stats.num_quadratic_objective_nz > 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : 
Objective sense      : Minimize
Variables            :       5  [Nneg: 1,  Binary: 1,  General Integer: 1,  Semi-continuous: 1,  Semi-integer: 1]
Objective nonzeros   :       0
Linear constraints   :       0
  Nonzeros           :       0
  RHS nonzeros       :       0

Variables            : Min LB: 0.000000         Max UB: 1.000000       
Objective nonzeros   : Min   : all zero         Max   : all zero       
Linear constraints   :
  Nonzeros           : Min   : all zero         Max   : all zero       
  RHS nonzeros       : Min   : all zero         Max   : all zero       
"""
        self.assertEqual(stats_str, expected_stats_str)


class GetStatsMultiObjTest(GetStatsTestCase):

    def testGetStats(self):
        cpx = self._newCplex()
        cpx.read(MULTIOBJ_EXAMPLE_FILE)
        stats = cpx.get_stats()
        self.checkAll(stats)

    def checkName(self, stats):
        self.assertEqual(stats.name, MULTIOBJ_EXAMPLE_FILE)

    def checkVariableData(self, stats):
        self.assertEqual(stats.num_variables, 25)
        self.assertEqual(stats.num_nonnegative, 0)
        self.assertEqual(stats.num_fixed, 0)
        self.assertEqual(stats.num_boxed, 7)
        self.assertEqual(stats.num_free, 0)
        self.assertEqual(stats.num_other, 0)
        self.assertEqual(stats.num_binary, 9)
        self.assertEqual(stats.num_integer, 9)
        self.assertEqual(stats.num_semicontinuous, 0)
        self.assertEqual(stats.num_semiinteger, 0)
        self.assertEqual(stats.num_quadratic_variables, 0)
        self.assertEqual(stats.num_linear_objective_nz, 18)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)

    def checkLinearConstraintData(self, stats):
        self.assertEqual(stats.num_linear_constraints, 7)
        self.assertEqual(stats.num_linear_less, 0)
        self.assertEqual(stats.num_linear_equal, 7)
        self.assertEqual(stats.num_linear_greater, 0)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertEqual(stats.num_linear_nz, 65)
        self.assertEqual(stats.num_linear_rhs_nz, 7)

    def checkIndicatorData(self, stats):
        self.assertEqual(stats.num_indicator_constraints, 9)
        self.assertEqual(stats.num_indicator_less, 9)
        self.assertEqual(stats.num_indicator_equal, 0)
        self.assertEqual(stats.num_indicator_greater, 0)
        self.assertEqual(stats.num_indicator_complemented, 9)
        self.assertEqual(stats.num_indicator_nz, 9)
        self.assertEqual(stats.num_indicator_rhs_nz, 0)

    def checkMinMaxVariableData(self, stats):
        self.assertEqual(stats.min_lower_bound, 0.0)
        self.assertEqual(stats.max_upper_bound, 9944.0)
        self.assertEqual(stats.min_linear_objective, 0.6)
        self.assertEqual(stats.max_linear_objective, 2.29)
        self.assertEqual(stats.num_quadratic_objective_nz, 0)
        self.assertFalse(hasattr(stats, 'min_quadratic_objective'))
        self.assertFalse(hasattr(stats, 'max_quadratic_objective'))

    def checkMinMaxLinearConstraintData(self, stats):
        self.assertEqual(stats.min_linear_constraints, 1.0)
        self.assertEqual(stats.max_linear_constraints, 510.0)
        self.assertEqual(stats.min_linear_constraints_rhs, 55.0)
        self.assertEqual(stats.max_linear_constraints_rhs, 2000.0)
        self.assertEqual(stats.num_linear_range, 0)
        self.assertFalse(hasattr(stats, 'min_linear_range'))
        self.assertFalse(hasattr(stats, 'max_linear_range'))

    def checkMinMaxIndicatorConstraintData(self, stats):
        self.assertEqual(stats.num_indicator_constraints, 9)
        self.assertEqual(stats.min_indicator, 1.0)
        self.assertEqual(stats.max_indicator, 1.0)
        self.assertEqual(stats.min_indicator_rhs, -CPX_INFBOUND)
        self.assertEqual(stats.max_indicator_rhs, CPX_INFBOUND)

    def checkString(self, stats):
        stats_str = str(stats)
        expected_stats_str = """\
Problem name         : ../../../examples/data/dietmultiobj.lp
Objective sense      : Minimize
Variables            :      25  [Box: 7,  Binary: 9,  General Integer: 9]
Objectives           :       2
  Objective nonzeros :      18
Linear constraints   :       7  [Equal: 7]
  Nonzeros           :      65
  RHS nonzeros       :       7
Indicator constraints:       9  [Less: 9]
  Complemented       :       9
  Nonzeros           :       9
  RHS nonzeros       :       0

Variables            : Min LB: 0.000000         Max UB: 9944.000       
Objective nonzeros   : Min   : 0.6000000        Max   : 2.290000       
Linear constraints   :
  Nonzeros           : Min   : 1.000000         Max   : 510.0000       
  RHS nonzeros       : Min   : 55.00000         Max   : 2000.000       
Indicator constraints:
  Nonzeros           : Min   : 1.000000         Max   : 1.000000       
  RHS nonzeros       : Min   : all zero         Max   : all zero       
"""
        self.assertEqual(stats_str, expected_stats_str)


def main():
    """The main function."""
    unittest.main()

if __name__ == "__main__":
    main()
