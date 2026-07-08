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
Tests solution methods on a simple model with PWL constraints.

No command line arguments are required.
"""
import unittest
import os
import errno
import cplex
import testemptysolution
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes


class PwlSolutionTests(testemptysolution.EmptySolutionTests):
    """Tests all of the SolutionInterface methods on a simple model."""

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        self.cpx.variables.add(names=['y', 'x'])
        self.cpx.pwl_constraints.add(vary='y',
                                     varx='x',
                                     preslope=0.5,
                                     postslope=2.0,
                                     breakx=[0.0, 1.0, 2.0],
                                     breaky=[0.0, 1.0, 4.0],
                                     name='pwl1')
        self.cpx.solve()

    def testGetStatus(self):
        self.assertEqual(self.cpx.solution.get_status(),
                         self.cpx.solution.status.MIP_optimal)

    def testGetMethod(self):
        self.assertEqual(self.cpx.solution.get_method(),
                         self.cpx.solution.method.MIP)

    def testGetStatusString(self):
        self.assertEqual(self.cpx.solution.get_status_string(),
                         'integer optimal solution')

    def testGetValues(self):
        self.assertEqual(self.cpx.solution.get_values(), [0.0, 0.0])

    def testGetReducedCosts(self):
        self.skipIfParamTesting(self.cpx)
        try:
            self.cpx.solution.get_reduced_costs()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_MIP)

    def testGetDualValues(self):
        self.skipIfParamTesting(self.cpx)
        try:
            self.cpx.solution.get_dual_values()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_MIP)

    def testGetQuadraticDualSlack(self):
        self.skipIfParamTesting(self.cpx)
        self.assertEqual(self.cpx.solution.get_quadratic_dualslack(), [])
        try:
            self.cpx.solution.get_quadratic_dualslack(0)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_MIP)

    def testGetIntegerQuality(self):
        self.skipIfParamTesting(self.cpx)
        qmlist = self._getQualityMetrics()
        # We simply test that the function can be called without an exception.
        # Kind of lame, but better than nothing.
        num_times_called = 0
        for qm in qmlist:
            try:
                self.cpx.solution.get_integer_quality(qm)
                num_times_called += 1
            except CplexSolverError as cse:
                if cse.args[2] not in (error_codes.CPXERR_NOT_FOR_MIP,
                                       error_codes.CPXERR_BAD_ARGUMENT,
                                       error_codes.CPXERR_NO_BARRIER_SOLN):
                    raise
        self.assertTrue(num_times_called)

    def testMaxPwlSlackInfeasFloat(self):
        actual = self.cpx.solution.get_float_quality(
            self.cpx.solution.quality_metric.max_pwl_slack_infeasibility)
        self.assertEqual(0.0, actual)

    def testMaxPwlSlackInfeasInt(self):
        actual = self.cpx.solution.get_integer_quality(
            self.cpx.solution.quality_metric.max_pwl_slack_infeasibility)
        self.assertEqual(-1, actual)

    def testMaxPwlSlackInfeas(self):
        actual = self.cpx.solution.get_float_quality(
            self.cpx.solution.quality_metric.sum_pwl_slack_infeasibility)
        self.assertEqual(0.0, actual)

    def testGetFloatQuality(self):
        self.skipIfParamTesting(self.cpx)
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
                                       error_codes.CPXERR_NO_BARRIER_SOLN,
                                       error_codes.CPXERR_NO_KAPPASTATS,
                                       error_codes.CPXERR_NOT_FOR_MIP):
                    raise
        self.assertTrue(num_times_called)

    def testGetSolutionType(self):
        self.assertEqual(self.cpx.solution.get_solution_type(),
                         self.cpx.solution.type.primal)

    def testGetQualityMetrics(self):
        quality_metrics = self.cpx.solution.get_quality_metrics()
        # If this fails then this test needs to be updated:
        self.assertEqual(len(quality_metrics.__dict__), 16)
        self.assertEqual(quality_metrics.error_Ax_b_max, 0)
        self.assertEqual(quality_metrics.error_Ax_b_total, 0)
        self.assertEqual(quality_metrics.integrality_error_max, 0)
        self.assertEqual(quality_metrics.integrality_error_total, 0)
        self.assertEqual(quality_metrics.objective, 0)
        self.assertEqual(quality_metrics.quality_type, 'MIP')
        self.assertEqual(quality_metrics.slack_bound_error_max, 0)
        self.assertEqual(quality_metrics.slack_bound_error_total, 0)
        self.assertEqual(quality_metrics.solver, 'MILP')
        self.assertEqual(quality_metrics.x_bound_error_max, 0)
        self.assertEqual(quality_metrics.x_bound_error_total, 0)
        self.assertEqual(quality_metrics.x_norm_max, 0)
        self.assertEqual(quality_metrics.x_norm_total, 0)
        self.assertEqual(quality_metrics.piecewise_linear_error_total, 0)
        self.assertEqual(quality_metrics.piecewise_linear_error_max, 0)
        expected_str = """\
Incumbent solution:
MILP objective                                 0.0000000000e+00
MILP solution norm |x| (Total, Max)            0.00000e+00  0.00000e+00
MILP solution error (Ax=b) (Total, Max)        0.00000e+00  0.00000e+00
MILP x bound error (Total, Max)                0.00000e+00  0.00000e+00
MILP x integrality error (Total, Max)          0.00000e+00  0.00000e+00
MILP slack bound error (Total, Max)            0.00000e+00  0.00000e+00
MILP piecewise linear error (Total, Max)       0.00000e+00  0.00000e+00
"""
        self.checkQualityMetricsString(str(quality_metrics), expected_str)

    def testGetBestObjective(self):
        self.assertEqual(self.cpx.solution.MIP.get_best_objective(), 0.0)

    def testGetCutoff(self):
        self.assertEqual(self.cpx.solution.MIP.get_cutoff(), 0.0)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
