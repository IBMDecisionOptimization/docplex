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
Tests solution methods on a simple LP model.

No command line arguments are required.
"""
import unittest
import os
import errno
import cplex
import testemptysolution
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes


class SimpleSolutionTests(testemptysolution.EmptySolutionTests):
    """Tests all of the SolutionInterface methods on a simple LP model."""

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        # Builds the simple model from lpex1.py.
        self.cpx.objective.set_sense(self.cpx.objective.sense.maximize)
        self.cpx.variables.add(obj=[1., 2., 3.],
                               ub=[40., cplex.infinity, cplex.infinity])
        self.cpx.linear_constraints.add(lin_expr=[[[0, 1, 2], [-1., 1., 1.]],
                                                  [[0, 1, 2], [1., -3., 1.]]],
                                        senses="LL",
                                        rhs=[20., 30.])
        self.cpx.solve()

    def testGetObjectiveValue(self):
        self.assertEqual(self.cpx.solution.get_objective_value(),
                         202.5)

    def testGetValues(self):
        self.assertEqual(self.cpx.solution.get_values(),
                         [40., 17.5, 42.5])

    def testGetReducedCosts(self):
        self.assertEqual(self.cpx.solution.get_reduced_costs(),
                         [3.5, 0.0, 0.0])

    def testGetDualValues(self):
        self.assertEqual(self.cpx.solution.get_dual_values(),
                         [2.75, 0.25])

    def testGetLinearSlacks(self):
        self.assertEqual(self.cpx.solution.get_linear_slacks(),
                         [0.0, 0.0])

    def testGetActivityLevels(self):
        self.assertEqual(self.cpx.solution.get_activity_levels(),
                         [20.0, 30.0])

    def testGetQualityMetrics(self):
        quality_metrics = self.cpx.solution.get_quality_metrics()
        # If this fails then this test needs to be updated:
        self.assertEqual(len(quality_metrics.__dict__), 12,
                         quality_metrics.__dict__)
        self.assertEqual(quality_metrics.quality_type, "simplex")
        self.assertFalse(quality_metrics.scaled)
        self.assertEqual(quality_metrics.max_x, 42.5)
        self.assertEqual(quality_metrics.max_pi, 2.75)
        self.assertEqual(quality_metrics.max_reduced_cost, 3.5)
        self.assertEqual(quality_metrics.max_bound_infeas, 0)
        self.assertEqual(quality_metrics.max_reduced_cost_infeas, 0)
        self.assertEqual(quality_metrics.max_Ax_minus_b, 0)
        self.assertEqual(quality_metrics.max_c_minus_Bpi, 0)
        self.assertEqual(quality_metrics.max_slack, 0)
        expected_str = """\
There are no bound infeasibilities.
There are no reduced-cost infeasibilities.
Maximum Ax-b residual              = 0
Maximum c-B'pi residual            = 0
Maximum |x|                        = 42.5
Maximum |slack|                    = 0
Maximum |pi|                       = 2.75
Maximum |red-cost|                 = 3.5
Condition number of unscaled basis = 3.0e+00
"""
        self.checkQualityMetricsString(str(quality_metrics), expected_str)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
