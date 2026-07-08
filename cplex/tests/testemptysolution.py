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
Tests solution methods on empty model.

No command line arguments are required.
"""
import unittest
import os
import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase

class EmptySolutionTests(CplexTestCase):
    """Tests all of the SolutionInterface methods on an empty solution."""

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        self.cpx.solve()

    def testGetStatus(self):
        self.assertEqual(self.cpx.solution.get_status(),
                         self.cpx.solution.status.optimal)

    def testGetMethod(self):
        self.assertEqual(self.cpx.solution.get_method(),
                         self.cpx.solution.method.dual)

    def testGetStatusString(self):
        self.assertEqual(self.cpx.solution.get_status_string(), 'optimal')

    def testGetObjectiveValue(self):
        self.assertEqual(self.cpx.solution.get_objective_value(), 0.0)

    def testGetValues(self):
        self.assertEqual(self.cpx.solution.get_values(), [])

    def testGetReducedCosts(self):
        self.assertEqual(self.cpx.solution.get_reduced_costs(), [])

    def testGetDualValues(self):
        self.assertEqual(self.cpx.solution.get_dual_values(), [])

    def testGetQuadraticDualSlack(self):
        self.assertEqual(self.cpx.solution.get_quadratic_dualslack(), [])
        try:
            self.cpx.solution.get_quadratic_dualslack(0)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_DUAL_SOLN)

    def testGetLinearSlacks(self):
        self.assertEqual(self.cpx.solution.get_linear_slacks(), [])

    def testGetIndicatorSlacks(self):
        self.assertEqual(self.cpx.solution.get_indicator_slacks(), [])

    def testGetQuadraticSlacks(self):
        self.assertEqual(self.cpx.solution.get_quadratic_slacks(), [])

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
                # CPXERR_BAD_ARGUMENT is for QualityMetric attributes that
                # should use get_float_quality.  The other two are expected
                # because the default is simplex for an empty model.
                if cse.args[2] not in (error_codes.CPXERR_BAD_ARGUMENT,
                                       error_codes.CPXERR_NOT_MIP,
                                       error_codes.CPXERR_NO_BARRIER_SOLN,):
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
                # These two errors are expected because the default is simplex
                # for an empty model.
                if cse.args[2] not in (error_codes.CPXERR_NOT_MIP,
                                       error_codes.CPXERR_NO_BARRIER_SOLN,):
                    raise
        self.assertTrue(num_times_called)

    def testGetSolutionType(self):
        self.assertEqual(self.cpx.solution.get_solution_type(),
                         self.cpx.solution.type.basic)

    def testIsPrimalFeasible(self):
        self.assertTrue(self.cpx.solution.is_primal_feasible())

    def testIsDualFeasible(self):
        self.assertTrue(self.cpx.solution.is_dual_feasible())

    def testGetActivityLevels(self):
        self.assertEqual(self.cpx.solution.get_activity_levels(), [])

    def testGetQuadraticActivityLevels(self):
        self.assertEqual(self.cpx.solution.get_quadratic_activity_levels(), [])

    def testGetQualityMetrics(self):
        quality_metrics = self.cpx.solution.get_quality_metrics()
        # If this fails then this test needs to be updated:
        self.assertEqual(len(quality_metrics.__dict__), 12,
                         quality_metrics.__dict__)
        self.assertEqual(quality_metrics.quality_type, "simplex")
        self.assertFalse(quality_metrics.scaled)
        self.assertEqual(quality_metrics.max_x, 0)
        self.assertEqual(quality_metrics.max_pi, 0)
        self.assertEqual(quality_metrics.max_reduced_cost, 0)
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
Maximum |x|                        = 0
Maximum |pi|                       = 0
Maximum |red-cost|                 = 0
Condition number of unscaled basis = 0.0e+00
"""
        self.checkQualityMetricsString(str(quality_metrics), expected_str)

    def checkQualityMetricsString(self, actual_str, expected_str):
        # Strip out any DEVINFO lines that might come while in DEBUG mode.
        tmp = []
        for line in actual_str.splitlines(True):
            if line.startswith("DEVINFO"):
                continue
            tmp.append(line)
        actual_str = "".join(tmp)
        self.assertEqual(actual_str, expected_str)

    def checkQualityMetricsSubStrings(self, actual_str, substrlist):
        for substr in substrlist:
            index = actual_str.find(substr)
            self.assertGreaterEqual(index, 0, """Failed to find substring!
actual_str: {0}
substring: {1}
""".format(actual_str, substr))

    def _getQualityMetrics(self):
        """
        Gets all of the attributes of the QualityMetric class.

        This is a hack to get all of the attributes of the custom QualityMetric
        class (ideally, it would be an iterator).
        """
        qmlist = []
        for key, value in cplex._internal._subinterfaces.QualityMetric.__dict__.items():
            if not key.startswith('__'):
                qmlist.append(value)
        self.assertTrue(len(qmlist) > 0)
        return qmlist

    def testGetBestObjective(self):
        self._checkMIPSolutionFunc(self.cpx.solution.MIP.get_best_objective)

    def testGetCutoff(self):
        self._checkMIPSolutionFunc(self.cpx.solution.MIP.get_cutoff)

    def testGetMIPRelativeGap(self):
        if self.cpx.get_problem_type() == self.cpx.problem_type.MILP:
            self.assertEqual(
                self.cpx.solution.MIP.get_mip_relative_gap(), 0.0)
        else:
            self._checkMIPSolutionFunc(
                self.cpx.solution.MIP.get_mip_relative_gap)

    def testGetIncumbentNode(self):
        if self.cpx.get_problem_type() == self.cpx.problem_type.MILP:
            self.assertGreaterEqual(
                self.cpx.solution.MIP.get_incumbent_node(), 0)
        else:
            # Rather than raise an error get_incumbent_node returns -1
            # if "no solution, problem, or environment exists".
            self.assertEqual(self.cpx.solution.MIP.get_incumbent_node(), -1)

    def testGetNumCuts(self):
        def func_with_arg():
            self.cpx.solution.MIP.get_num_cuts(
                self.cpx.solution.MIP.cut_type.zero_half)
        if self.cpx.get_problem_type() == self.cpx.problem_type.MILP:
            self._checkMIPGetNumCuts()
        else:
            self._checkMIPSolutionFunc(func_with_arg)

    def testGetSubproblemStatus(self):
        # Rather than raise an error get_subproblem_status returns 0 if
        # no solution exists.  A nonzero return value reports that there
        # was an error termination where a subproblem could not be solved
        # to completion.
        self.assertEqual(self.cpx.solution.MIP.get_subproblem_status(), 0)

    def _checkMIPGetNumCuts(self):
        cut_type = self.cpx.solution.MIP.cut_type
        for which in cut_type:
            self.assertGreaterEqual(
                self.cpx.solution.MIP.get_num_cuts(which), 0)

    def _checkMIPSolutionFunc(self, func):
        try:
            func()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_MIP)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
