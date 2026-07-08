# -*- coding: utf-8 -*-
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
Tests the Cplex.set_problem_type method with solution pool argument.

The function just changes the problem type based on a solution pool element
and then solves the fixed problem.

After that it queries the primal and dual solution and makes sure
complementary slackness is satisfied.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase

MILP_EXAMPLE_FILE = '../../../examples/data/caso8.mps'
MIQP_EXAMPLE_FILE = '../../../examples/data/al970121_miqp.sav.gz'

# Minimal number of solutions the model must produce.  Note that the first
# thing we test is the incumbent solution which has index -1.  Since -1 is an
# invalid value for problem types we can be sure that the probtype and soln
# arguments are not swapped (which was the issue in RTC-21794).
MIN_SOLS = 5

EPSZERO = 1e-6

class SetProblemTypeTests(CplexTestCase):

    def testMilp(self):
        self._testModel(MILP_EXAMPLE_FILE)

    def testMiqp(self):
        self._testModel(MIQP_EXAMPLE_FILE)

    def _testModel(self, modelpath):
        for s in range(-1, MIN_SOLS):
            cpx = self._newCplex()
            cpx.read(modelpath)
            cols = cpx.variables.get_num()
            rows = cpx.linear_constraints.get_num()
            cpx.parameters.mip.limits.solutions.set(MIN_SOLS)
            cpx.parameters.emphasis.numerical.set(
                cpx.parameters.emphasis.numerical.values.on)
            cpx.solve()
            solns = cpx.solution.pool.get_num()
            self.assertFalse(
                solns < MIN_SOLS,
                "Not enough solutions, expected {0} but got only {1}"
                .format(MIN_SOLS, solns))
            probtype = cpx.get_problem_type()
            if probtype == cpx.problem_type.MILP:
                newprobtype = cpx.problem_type.fixed_MILP
            elif probtype == cpx.problem_type.MIQP:
                newprobtype = cpx.problem_type.fixed_MIQP
            else:
                self.fail("Invalid problem type!")
            cpx.set_problem_type(newprobtype, s)
            cpx.solve()
            self.assertEqual(cpx.solution.get_status(),
                             cpx.solution.status.optimal)

            # Test complementary slackness of duals (rows).
            slack = cpx.solution.get_linear_slacks()
            pi = cpx.solution.get_dual_values()
            for i in range(rows):
                self.assertAlmostEqual(
                    slack[i] * pi[i], 0.0, places=6,
                    msg="No complementary slackness for row {0}".format(i))

            # Test complementary slackness of duals (columns).
            x = cpx.solution.get_values()
            dj = cpx.solution.get_reduced_costs()
            lb = cpx.variables.get_lower_bounds()
            ub = cpx.variables.get_upper_bounds()
            for j in range(cols):
                atlb = lb[j] > -cplex.infinity and abs(x[j] - lb[j]) < EPSZERO
                atub = ub[j] < cplex.infinity and abs(x[j] - ub[j]) < EPSZERO
                if abs(dj[j]) > EPSZERO:
                    # Non-zero dj -> variable must be at a bound.
                    self.assertTrue(
                        atlb or atub,
                        "No complementary slackness for x[{0}]".format(j))
                elif not (atlb or atub):
                    # Variable is not at bound -> dj must be zero.
                    self.assertAlmostEqual(
                        dj[j], 0.0, places=6,
                        msg="No complementary slackness for x[{0}]".format(j))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
