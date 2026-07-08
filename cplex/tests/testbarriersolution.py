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
Tests solution methods on a simple LP model (using barrier).

No command line arguments are required.
"""
import unittest
import os
import errno
import cplex
import testemptysolution
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes


class BarrierSolutionTests(testemptysolution.EmptySolutionTests):
    """Tests all of the SolutionInterface methods on a simple LP model."""

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        self.cpx.read("../../data/qpex.lp")
        self.cpx.solve()

    def testGetMethod(self):
        self.assertEqual(self.cpx.solution.get_method(),
                         self.cpx.solution.method.barrier)

    def testGetObjectiveValue(self):
        self.assertAlmostEqual(self.cpx.solution.get_objective_value(),
                               2.0156165232891574)

    def testGetValues(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_values(),
                                    [0.13911493520482582,
                                     0.5984654742056659,
                                     0.8983957232479367])

    def testGetReducedCosts(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_reduced_costs(),
                                    [-1.647665492932049e-08,
                                     -3.843456664043288e-09,
                                     -2.5603155151543433e-09])

    def testGetDualValues(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_dual_values(),
                                    [1.2338512374765798e-10,
                                     7.478331894313442e-11])

    def testGetLinearSlacks(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_linear_slacks(),
                                    [18.642253737751222, 30.757885764164236])

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
                                       error_codes.CPXERR_NOT_MIP,):
                    raise
        self.assertTrue(num_times_called)

    def testGetSolutionType(self):
        self.assertEqual(self.cpx.solution.get_solution_type(),
                         self.cpx.solution.type.nonbasic)

    def testGetActivityLevels(self):
        self.assertListsAlmostEqual(self.cpx.solution.get_activity_levels(),
                                    [1.357746262248778, -0.7578857641642358])

    def testGetQualityMetrics(self):
        quality_metrics = self.cpx.solution.get_quality_metrics()
        # If this fails then this test needs to be updated:
        self.assertEqual(len(quality_metrics.__dict__), 28,
                         quality_metrics.__dict__)
        self.assertAlmostEqual(quality_metrics.primal_objective,
                               2.0156165232891574)
        self.assertAlmostEqual(quality_metrics.dual_error_max,
                               8.881784197001252e-16)
        self.assertAlmostEqual(quality_metrics.dual_normalized_error,
                               9.55030558817339e-18)
        self.assertAlmostEqual(quality_metrics.row_complementarity_total,
                               4.600353565485662e-09)
        self.assertEqual(quality_metrics.quality_type, 'barrier')
        self.assertEqual(quality_metrics.primal_x_bound_error_max, 0.0)
        self.assertEqual(quality_metrics.dual_pi_bound_error_max, 0.0)
        self.assertEqual(quality_metrics.primal_slack_bound_error_max, 0.0)
        self.assertEqual(quality_metrics.primal_x_bound_error_total, 0.0)
        self.assertAlmostEqual(quality_metrics.column_complementarity_total,
                               6.892501406900286e-09)
        self.assertAlmostEqual(quality_metrics.dual_norm_total,
                               2.288042710851812e-08)
        self.assertAlmostEqual(quality_metrics.row_complementarity_max,
                               2.300176784367674e-09)
        self.assertAlmostEqual(quality_metrics.complementarity_total,
                               1.1492854972385948e-08)
        self.assertEqual(quality_metrics.dual_pi_bound_error_total, 0.0)
        self.assertAlmostEqual(quality_metrics.column_complementarity_max,
                               2.30017650898e-09)
        self.assertAlmostEqual(quality_metrics.dual_objective,
                               2.015616534782012)
        self.assertEqual(quality_metrics.dual_reduced_cost_bound_error_total,
                         0.0)
        self.assertAlmostEqual(quality_metrics.duality_gap,
                               -1.149285466794936e-08)
        self.assertEqual(quality_metrics.dual_reduced_cost_bound_error_max, 0.0)
        self.assertAlmostEqual(quality_metrics.primal_norm_total,
                               1.6359761326584283)
        self.assertAlmostEqual(quality_metrics.dual_error_total,
                               1.3322676295501878e-15)
        self.assertAlmostEqual(quality_metrics.primal_error_total,
                               1.887379141862766e-15)
        self.assertAlmostEqual(quality_metrics.primal_normalized_error,
                               2.756384846596202e-17)
        self.assertAlmostEqual(quality_metrics.primal_error_max,
                               1.1657341758564144e-15)
        self.assertEqual(quality_metrics.primal_slack_bound_error_total, 0.0)
        self.assertAlmostEqual(quality_metrics.dual_norm_max,
                               1.647665492932049e-08)
        self.assertAlmostEqual(quality_metrics.primal_norm_max,
                               0.8983957232479367)
        expected_strings = ["Primal objective",
                            "Dual objective",
                            "Duality gap",
                            "Complementarity (Total)",
                            "Column complementarity (Total, Max)",
                            "Row complementarity (Total, Max)",
                            "Primal norm |x| (Total, Max)",
                            "Dual norm |rc| (Total, Max)",
                            "Primal error (Ax=b) (Total, Max)",
                            "Dual error (A'pi+rc=c) (Total, Max)",
                            "Primal x bound error (Total, Max)",
                            "Primal slack bound error (Total, Max)",
                            "Dual pi bound error (Total, Max)",
                            "Dual rc bound error (Total, Max)",
                            "Primal normalized error (Ax=b) (Max)",
                            "Dual normalized error (A'pi+rc=c) (Max)"]
        self.checkQualityMetricsSubStrings(str(quality_metrics),
                                           expected_strings)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
