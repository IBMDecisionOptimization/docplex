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
Tests solution methods on a simple MIP model.

No command line arguments are required.
"""
import unittest
import os
import errno
import cplex
import testemptysolution
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes

EPSILON = 1e-6


class FeasoptSolutionTests(testemptysolution.EmptySolutionTests):
    """Tests all of the SolutionInterface methods on a simple MIP model."""

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        self.cpx.read("../../data/inflp.mps")
        self.cpx.feasopt(self.cpx.feasopt.all_constraints())

    def testIsPrimalFeasible(self):
        self.assertFalse(self.cpx.solution.is_primal_feasible())

    def testIsDualFeasible(self):
        self.assertFalse(self.cpx.solution.is_dual_feasible())

    def testGetStatus(self):
        self.assertEqual(self.cpx.solution.get_status(),
                         self.cpx.solution.status.feasible_relaxed_sum)

    def testGetMethod(self):
        self.assertEqual(self.cpx.solution.get_method(),
                         self.cpx.solution.method.feasopt)

    def testGetStatusString(self):
        self.assertEqual(self.cpx.solution.get_status_string(),
                         'feasible relaxed sum of infeasibilities')

    def testGetObjectiveValue(self):
        self.assertAlmostEqual(self.cpx.solution.get_objective_value(),
                               129.39394461854306)

    def testGetValues(self):
        # We only check that we get the expected number of values.
        self.assertEqual(len(self.cpx.solution.get_values()), 54)

    def testGetReducedCosts(self):
        try:
            self.cpx.solution.get_reduced_costs()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_DUAL_SOLN)

    def testGetDualValues(self):
        try:
            self.cpx.solution.get_dual_values()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_DUAL_SOLN)

    def testGetLinearSlacks(self):
        # We only check that we get the expected number of values.
        self.assertEqual(len(self.cpx.solution.get_linear_slacks()), 477)

    def testGetIntegerQuality(self):
        qmlist = self._getQualityMetrics()
        # We simply test that the function can be called without an exception.
        # Kind of lame, but better than nothing.
        num_times_called = 0
        for qm in qmlist:
            try:
                self.cpx.solution.get_integer_quality(qm)
                num_times_called += 1
            except CplexSolverError as cse:
                if cse.args[2] not in (error_codes.CPXERR_NO_DUAL_SOLN,
                                       error_codes.CPXERR_BAD_ARGUMENT,
                                       error_codes.CPXERR_NOT_MIP,
                                       error_codes.CPXERR_NO_BARRIER_SOLN):
                    raise
        self.assertTrue(num_times_called)

    def testGetFloatQuality(self):
        qmlist = self._getQualityMetrics()
        # We simply test that the function can be called without an exception.
        # Kind of lame, but better than nothing.
        num_times_called = 0
        for qm in qmlist:
            try:
                self.cpx.solution.get_float_quality(qm)
                num_times_called += 1
            except CplexSolverError as cse:
                if cse.args[2] not in (error_codes.CPXERR_NO_LU_FACTOR,
                                       error_codes.CPXERR_NO_DUAL_SOLN,
                                       error_codes.CPXERR_NOT_MIP,
                                       error_codes.CPXERR_NO_BARRIER_SOLN,):
                    raise
        self.assertTrue(num_times_called)

    def testGetSolutionType(self):
        self.assertEqual(self.cpx.solution.get_solution_type(),
                         self.cpx.solution.type.primal)

    def testGetActivityLevels(self):
        # We only check that we get the expected number of values.
        self.assertEqual(len(self.cpx.solution.get_activity_levels()), 477)

    def testGetQualityMetrics(self):
        quality_metrics = self.cpx.solution.get_quality_metrics()
        # If this fails then this test needs to be updated:
        self.assertEqual(len(quality_metrics.__dict__), 11,
                         quality_metrics.__dict__)
        self.assertEqual(quality_metrics.quality_type, 'feasopt')
        self.assertAlmostEqual(quality_metrics.max_scaled_x, 73794.99155842418,
                               delta=EPSILON)
        self.assertTrue(quality_metrics.scaled)
        self.assertAlmostEqual(quality_metrics.max_Ax_minus_b,
                               1.2871623766841367e-09)
        self.assertAlmostEqual(quality_metrics.max_bound_infeas,
                               96.67724570192809)
        self.assertAlmostEqual(quality_metrics.max_slack,
                               48192.64754835691,
                               delta=EPSILON)
        self.assertAlmostEqual(quality_metrics.max_scaled_bound_infeas,
                               9.111506922349253)
        self.assertAlmostEqual(quality_metrics.max_scaled_Ax_minus_b,
                               3.637978807091713e-12)
        self.assertAlmostEqual(quality_metrics.max_scaled_slack,
                               12048.161887090482, delta=EPSILON)
        self.assertAlmostEqual(quality_metrics.max_x,
                               73794.99155842418, delta=EPSILON)
        expected_strings = ["Max. unscaled (scaled) bound infeas.",
                            "Max. unscaled (scaled) Ax-b resid.",
                            "Max. unscaled (scaled) |x|",
                            "Max. unscaled (scaled) |slack|"]
        self.checkQualityMetricsSubStrings(str(quality_metrics),
                                           expected_strings)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
