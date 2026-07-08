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
Tests solution methods on simple QCP model.

No command line arguments are required.
"""
import unittest
import os
import errno
import cplex
import testemptysolution
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplex import SparsePair
from cplex import SparseTriple

# We expect the objective to be more accurate
OBJECTIVE_EPSILON = 1e-7 # 1e-5
# and the reported errors should be small as well
QUALITY_EPSILON   = 1e-8
# For the other values here, we shouldn't be checking with such high
# accuracy (see RTC-36927).
VALUE_EPSILON   = 1e-4
SLACK_EPSILON   = 1e-3


class QPSolutionTests(testemptysolution.EmptySolutionTests):
    """Tests all of the SolutionInterface methods on a simple QCP model."""

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        # Builds the simple model from qcpex1.py.
        self.cpx.objective.set_sense(self.cpx.objective.sense.maximize)
        self.cpx.linear_constraints.add(rhs=[20., 30.], senses="LL")
        self.cpx.variables.add(obj=[1., 2., 3.],
                               ub=[40., cplex.infinity, cplex.infinity],
                               columns=[[[0,1],[-1.0, 1.0]],
                                        [[0,1],[ 1.0,-3.0]],
                                        [[0,1],[ 1.0, 1.0]]])
        self.cpx.objective.set_quadratic([[[0,1],[-33.0, 6.0]],
                                          [[0,1,2],[ 6.0,-22.0, 11.5]],
                                          [[1,2],[ 11.5, -11.0]]])
        Q = cplex.SparseTriple(ind1=[0, 1, 2],
                               ind2=[0, 1, 2],
                               val=[1.0] * 3)
        self.cpx.quadratic_constraints.add(rhs=1., quad_expr=Q)
        self.cpx.solve()

    def testGetMethod(self):
        self.assertEqual(self.cpx.solution.get_method(),
                         self.cpx.solution.method.barrier)

    def testGetObjectiveValue(self):
        self.assertAlmostEqual(self.cpx.solution.get_objective_value(),
                               2.002346650238365,
                               delta=OBJECTIVE_EPSILON)
    def testGetValues(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_values(),
                                    [0.1291194359137942,
                                     0.5499506118604034,
                                     0.8251560432906793],
                                    delta=VALUE_EPSILON)

    def testGetReducedCosts(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_reduced_costs(),
                                    [-3.783081683003068e-12,
                                     -8.390263881131885e-13,
                                     -5.422758396158515e-13],
                                    delta=VALUE_EPSILON)

    def testGetDualValues(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_dual_values(),
                                    [-3.4120721294412709e-09,
                                     -4.6098657898953302e-09],
                                    delta=VALUE_EPSILON)

    def testGetQuadraticDualSlack(self):
        def checkSparsePairs(lhs, rhs):
            size = len(lhs.ind)
            self.assertEqual(size, len(rhs.ind))
            for i in range(size):
                self.assertEqual(lhs.ind[i], rhs.ind[i])
                self.assertAlmostEqual(lhs.val[i], rhs.val[i], delta=SLACK_EPSILON)
        expected = SparsePair(ind=[0, 1, 2],
                              val=[0.03876243501879761,
                                   0.16509845778655804,
                                   0.24771676708851348])
        actuallst = self.cpx.solution.get_quadratic_dualslack()
        self.assertEqual(len(actuallst), 1)
        actual = actuallst[0]
        checkSparsePairs(actual, expected)
        actual = self.cpx.solution.get_quadratic_dualslack(0)
        checkSparsePairs(actual, expected)

    def testGetLinearSlacks(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_linear_slacks(),
                                    [18.75401278076271,
                                     30.695576356376737],
                                    delta=SLACK_EPSILON)

    def testGetQuadraticSlacks(self):
        self.assertListsAlmostEqual(
            self.cpx.solution.get_quadratic_slacks(),
            [4.5421444383464404e-12],
            delta=SLACK_EPSILON)

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
                if cse.args[2] not in (error_codes.CPXERR_NOT_FOR_QCP,
                                       error_codes.CPXERR_BAD_ARGUMENT,
                                       error_codes.CPXERR_NOT_MIP):
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
                                       error_codes.CPXERR_NOT_MIP,
                                       error_codes.CPXERR_NOT_FOR_QCP):
                    raise
        self.assertTrue(num_times_called)

    def testGetSolutionType(self):
        self.assertEqual(self.cpx.solution.get_solution_type(),
                         self.cpx.solution.type.nonbasic)

    def testGetActivityLevels(self):
        self.assertListsAlmostEqual(
            self.cpx.solution.get_activity_levels(),
            [1.2459872192372892, -0.695576356376737],
            delta=VALUE_EPSILON)

    def testGetQuadraticActivityLevels(self):
        self.assertListsAlmostEqual(
            self.cpx.solution.get_quadratic_activity_levels(),
            [0.9999999999954579],
            delta=VALUE_EPSILON)

    def testGetQualityMetrics(self):
        quality_metrics = self.cpx.solution.get_quality_metrics()
        # If this fails then this test needs to be updated:
        self.assertEqual(len(quality_metrics.__dict__), 16,
                         quality_metrics.__dict__)
        self.assertAlmostEqual(quality_metrics.error_Ax_b_max, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.error_Ax_b_total, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.error_xQx_dx_f_max, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.error_xQx_dx_f_total, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.norm_max,
                               0.8251560432906793,
                               delta=VALUE_EPSILON)
        self.assertAlmostEqual(quality_metrics.norm_total,
                               1.5042260910648768,
                               delta=VALUE_EPSILON)
        self.assertAlmostEqual(quality_metrics.normalized_error_max, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.objective,
                               2.002346650238365,
                               delta=OBJECTIVE_EPSILON)
        self.assertAlmostEqual(
            quality_metrics.quadratic_slack_bound_error_max, 0,
            delta=QUALITY_EPSILON)
        self.assertAlmostEqual(
            quality_metrics.quadratic_slack_bound_error_total, 0,
            delta=QUALITY_EPSILON)
        self.assertEqual(quality_metrics.quality_type,
                         'quadratically_constrained')
        self.assertAlmostEqual(quality_metrics.slack_bound_error_max, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.slack_bound_error_total, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.x_bound_error_max, 0,
                               delta=QUALITY_EPSILON)
        self.assertAlmostEqual(quality_metrics.x_bound_error_total, 0,
                               delta=QUALITY_EPSILON)
        expected_strings = ["Primal objective",
                            "Primal norm |x| (Total, Max)",
                            "Primal error (Ax=b) (Total, Max)",
                            "Primal error (x'Qx+dx=f) (Total, Max)",
                            "Primal x bound error (Total, Max)",
                            "Primal slack bound error (Total, Max)",
                            "Primal quad. slack bound error (Total, Max)",
                            "Primal normalized error (Ax=b) (Max)"]
        self.checkQualityMetricsSubStrings(str(quality_metrics),
                                           expected_strings)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
