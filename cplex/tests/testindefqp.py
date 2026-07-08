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
Tests indefinite QP problem.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase

EPSILON = 1.0e-8


class IndefQPTests(CplexTestCase):

    def testSimpleModel(self):
        cpx = self._newCplex()
        cpx.variables.add(lb=[-1.0, 0.0], ub=[1.0, 1.0])
        cpx.objective.set_quadratic([[[0, 1], [-3.0, -0.5]],
                                     [[0, 1], [-0.5, -3.0]]])
        cpx.linear_constraints.add(lin_expr=[[[0, 1], [ 1.0, 1.0]],
                                             [[0, 1], [-1.0, 1.0]]],
                                   rhs=[0.0, 0.0],
                                   senses=['G', 'G'])
        cpx.parameters.optimalitytarget.set(
            cpx.parameters.optimalitytarget.values.first_order)
        # may get either optimum
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.first_order)
        # force to (1, 1)
        cpx.linear_constraints.add(lin_expr=[[[0], [1.0]]],
                                   rhs=[-0.1],
                                   senses='G')
        cpx.solve()
        obj_val = cpx.solution.get_objective_value()
        values = cpx.solution.get_values()
        self.assertTrue(abs(obj_val - -3.5) < EPSILON)
        self.assertTrue(abs(values[0] - 1.0) < EPSILON)
        self.assertTrue(abs(values[1] - 1.0) < EPSILON)
        # force to (-1, 1)
        cpx.linear_constraints.set_senses(2, 'L')
        cpx.solve()
        obj_val = cpx.solution.get_objective_value()
        values = cpx.solution.get_values()
        self.assertTrue(abs(obj_val - -2.5) < EPSILON)
        self.assertTrue(abs(values[0] - -1.0) < EPSILON)
        self.assertTrue(abs(values[1] -  1.0) < EPSILON)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
