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


class MipSolutionTests(testemptysolution.EmptySolutionTests):
    """Tests all of the SolutionInterface methods on a simple MIP model."""

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        # Builds the MIP model from mipex1.py.
        self.cpx.objective.set_sense(self.cpx.objective.sense.maximize)
        self.cpx.variables.add(obj=[1., 2., 3., 1.],
                               lb=[0., 0., 0., 2.],
                               ub=[40., cplex.infinity, cplex.infinity, 3.],
                               types="CCCI")
        self.cpx.linear_constraints.add(lin_expr=[[[0, 1, 2, 3],
                                                   [-1., 1., 1., 10.]],
                                                  [[0, 1, 2],
                                                   [1., -3., 1.]],
                                                  [[1, 3], [1., -3.5]]],
                                        senses="LLE",
                                        rhs=[20., 30., 0.])
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

    def testGetObjectiveValue(self):
        self.assertEqual(self.cpx.solution.get_objective_value(),
                         122.5)

    def testGetValues(self):
        self.assertEqual(self.cpx.solution.get_values(),
                         [40.0, 10.5, 19.5, 3.0])

    def testGetReducedCosts(self):
        try:
            self.cpx.solution.get_reduced_costs()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_MIP)

    def testGetDualValues(self):
        try:
            self.cpx.solution.get_dual_values()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_MIP)

    def testGetQuadraticDualSlack(self):
        self.assertEqual(self.cpx.solution.get_quadratic_dualslack(), [])
        try:
            self.cpx.solution.get_quadratic_dualslack(0)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_MIP)

    def testGetLinearSlacks(self):
        self.assertEqual(self.cpx.solution.get_linear_slacks(),
                         [0.0, 2.0, 0.0])

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
                if cse.args[2] not in (error_codes.CPXERR_NOT_FOR_MIP,
                                       error_codes.CPXERR_BAD_ARGUMENT,
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
                                       error_codes.CPXERR_NOT_FOR_MIP,
                                       error_codes.CPXERR_NO_BARRIER_SOLN,
                                       error_codes.CPXERR_NO_KAPPASTATS):
                    raise
        self.assertTrue(num_times_called)

    def testGetSolutionType(self):
        self.assertEqual(self.cpx.solution.get_solution_type(),
                         self.cpx.solution.type.primal)

    def testGetActivityLevels(self):
        self.assertEqual(self.cpx.solution.get_activity_levels(),
                         [20.0, 28.0, 0.0])

    def testGetQualityMetrics(self):
        quality_metrics = self.cpx.solution.get_quality_metrics()
        # If this fails then this test needs to be updated:
        self.assertEqual(len(quality_metrics.__dict__), 14,
                         quality_metrics.__dict__)
        self.assertEqual(quality_metrics.error_Ax_b_max, 0)
        self.assertEqual(quality_metrics.error_Ax_b_total, 0)
        self.assertEqual(quality_metrics.integrality_error_max, 0)
        self.assertEqual(quality_metrics.integrality_error_total, 0)
        self.assertEqual(quality_metrics.objective, 122.5)
        self.assertEqual(quality_metrics.quality_type, 'MIP')
        self.assertEqual(quality_metrics.slack_bound_error_max, 0)
        self.assertEqual(quality_metrics.slack_bound_error_total, 0)
        self.assertEqual(quality_metrics.solver, 'MILP')
        self.assertEqual(quality_metrics.x_bound_error_max, 0)
        self.assertEqual(quality_metrics.x_bound_error_total, 0)
        self.assertEqual(quality_metrics.x_norm_max, 40.0)
        self.assertEqual(quality_metrics.x_norm_total, 73.0)
        expected_str = """\
Incumbent solution:
MILP objective                                 1.2250000000e+02
MILP solution norm |x| (Total, Max)            7.30000e+01  4.00000e+01
MILP solution error (Ax=b) (Total, Max)        0.00000e+00  0.00000e+00
MILP x bound error (Total, Max)                0.00000e+00  0.00000e+00
MILP x integrality error (Total, Max)          0.00000e+00  0.00000e+00
MILP slack bound error (Total, Max)            0.00000e+00  0.00000e+00
"""
        self.checkQualityMetricsString(str(quality_metrics), expected_str)

    def testGetBestObjective(self):
        self.assertEqual(self.cpx.solution.MIP.get_best_objective(),
                         122.5)

    def testGetCutoff(self):
        self.assertEqual(self.cpx.solution.MIP.get_cutoff(), 122.5)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
