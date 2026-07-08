# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Tests feasopt.

No command line arguments are required.
"""
import unittest
import cplex
from cplex.exceptions import (CplexSolverError,
                              WrongNumberOfArgumentsError,
                              error_codes)
from cplextestcase import CplexTestCase
from cplex._internal._subinterfaces import SolutionStatus


class FeasoptTests(CplexTestCase):
    """Test on empty model."""

    model = None

    # Expected status codes
    all_stat = SolutionStatus.feasible
    ub_stat = SolutionStatus.feasible
    lb_stat = SolutionStatus.feasible
    lin_stat = SolutionStatus.feasible
    quad_stat = SolutionStatus.feasible
    ind_stat = SolutionStatus.feasible

    def setUp(self):
        self.cpx = self._newCplex()
        if self.model:
            self.cpx.read(self.model)

    def testNoArgs(self):
        try:
            self.cpx.feasopt()
            self.fail()
        except WrongNumberOfArgumentsError:
            pass

    def testAllConstraints(self):
        self.cpx.feasopt(self.cpx.feasopt.all_constraints())
        self.assertEqual(self.cpx.solution.get_status(), self.all_stat)

    def testUBConstraints(self):
        self.cpx.feasopt(self.cpx.feasopt.upper_bound_constraints())
        self.assertEqual(self.cpx.solution.get_status(), self.ub_stat)

    def testLBConstraints(self):
        self.cpx.feasopt(self.cpx.feasopt.lower_bound_constraints())
        self.assertEqual(self.cpx.solution.get_status(), self.lb_stat)

    def testLinearConstraints(self):
        self.cpx.feasopt(self.cpx.feasopt.linear_constraints())
        self.assertEqual(self.cpx.solution.get_status(), self.lin_stat)

    def testQuadraticConstraints(self):
        self.cpx.feasopt(self.cpx.feasopt.quadratic_constraints())
        self.assertEqual(self.cpx.solution.get_status(), self.quad_stat)

    def testIndicatorConstraints(self):
        self.cpx.feasopt(self.cpx.feasopt.indicator_constraints())
        self.assertEqual(self.cpx.solution.get_status(), self.ind_stat)


class SimpleFeasoptTests(FeasoptTests):
    """Test on simple feasible LP."""

    model = "../../data/lpprog.lp"


class InfeasFeasoptTests(FeasoptTests):
    """Test on simple infeasible MILP."""

    model = "../../data/infeasible.lp"

    all_stat = SolutionStatus.MIP_feasible_relaxed_sum
    ub_stat = SolutionStatus.MIP_feasible_relaxed_sum
    lb_stat = SolutionStatus.MIP_feasible_relaxed_sum
    lin_stat = SolutionStatus.MIP_feasible_relaxed_sum
    quad_stat = SolutionStatus.MIP_infeasible
    ind_stat = SolutionStatus.MIP_infeasible


def main():
    unittest.main()

if __name__ == '__main__':
    main()
