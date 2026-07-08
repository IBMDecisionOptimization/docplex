
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
Tests boxQP problems.

Use various bound settings
"""
import unittest
from random import random, seed
from cplextestcase import CplexTestCase

EPSILON = 1.0e-8

                     


class BoxQPTests(CplexTestCase):

    def cont2binpresolve(self, orig):
        ''' Test the presolve reduction that transforms unconstrained
        continuous quadratic variables into binary.
        We want to test model with and without the reduction and test that
        we get the same objective value.'''

        # Acceptable optimization statuses
        opt_statuses = (orig.solution.status.MIP_optimal,
                        orig.solution.status.optimal_tolerance)

        # To store objective value
        objval = list()

        # Solve the model twice. The first time we disable the reduction
        # with parameter
        for k in range(2):
            with self._newCplex(orig) as cpx:
                cpx.parameters.optimalitytarget.set(
                    cpx.parameters.optimalitytarget.values.optimal_global)
                if k == 0:
                    # Is there a more robust way to do this?
                    cpx.parameters._set(2200, 1099511627776)
                cpx.solve()
                if k == 1:
                    # if the following assertion fails the test probably does nothing.
                    self.assertTrue( cpx.solution.progress.get_num_iterations() > 0)
                status = cpx.solution.get_status()
                self.assertTrue( status in opt_statuses )
                objval.append(cpx.solution.get_objective_value())

        self.assertAlmostEqual(objval[0], objval[1], delta=1e-5*(max([abs(v) for v in objval]) + 1.))


    def testBoxQp(self):
        ''' Test solving boxQP with various bounds as input.
        The reduction will transform continuous variables to binaries.
        Originally models have bounds 0 and 1 so it's very simple.
        We want to test everything is ok with various different types
        of non-standard bounds'''
        # first test the original model this has 
        with self._newCplex() as cpx:
            cpx.read('../../data/bglnqp-20-90-3.mps.gz')
            self.cont2binpresolve(cpx)

    def testBoxQpRange(self):
        '''Now set upper bounds to 2. This will test that range different
         from 1 are dealt with'''
        with self._newCplex() as cpx:
            cpx.read('../../data/bglnqp-20-90-3.mps.gz')
            nvars = cpx.variables.get_num()
            cpx.variables.set_upper_bounds(list(zip(range(nvars), [2.]*nvars)))
            self.cont2binpresolve(cpx)

    def testBoxQpLower(self):
        '''Now set upper bounds to 0 and lower bound to -1.
        This will test that nonzero lower bounds are dealt with
        (even if range is wrong)'''
        with self._newCplex() as cpx:
            cpx.read('../../data/bglnqp-20-90-3.mps.gz')
            nvars = cpx.variables.get_num()
            cpx.variables.set_upper_bounds(list(zip(range(nvars), [0.]*nvars)))
            cpx.variables.set_lower_bounds(list(zip(range(nvars), [-1.]*nvars)))
            self.cont2binpresolve(cpx)

    def testBoxQpBoth(self):
        '''Now set upper bounds to lower bound to -1.
        This will test that nonzero lower bounds are dealt with
        (even if range is wrong)'''
        with self._newCplex() as cpx:
            cpx.read('../../data/bglnqp-20-90-3.mps.gz')
            nvars = cpx.variables.get_num()
            cpx.variables.set_lower_bounds(list(zip(range(nvars), [-1.]*nvars)))
            self.cont2binpresolve(cpx)

    def testBoxQpRand(self):
        '''Now set some random bounds between -1 and 1'''
        with self._newCplex() as cpx:
            cpx.read('../../data/bglnqp-20-90-3.mps.gz')
            nvars = cpx.variables.get_num()
            seed(1)
            lb = [ -random()/4. -0.75 for i in range(nvars)]
            ub = [ -lb[i] for i in range(nvars)]
            cpx.variables.set_lower_bounds(list(zip(range(nvars), lb)))
            cpx.variables.set_upper_bounds(list(zip(range(nvars), ub)))
            self.cont2binpresolve(cpx)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
