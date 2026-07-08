# -*- coding: utf-8 -*-
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
Tests a simple model.

No command line arguments are required.
"""
import unittest
import os.path
import cplex
from cplextestcase import CplexTestCase


class SimpleModelTests(CplexTestCase):

    def testSimple(self):
        model = self._newCplex()
        my_obj = [1, 3, 6.24, 0.1]
        my_ub = [cplex.infinity, cplex.infinity, cplex.infinity, 48.98]
        my_lb = [28.6, -cplex.infinity, -cplex.infinity, 18]
        my_colnames = ["COLONE", "COLTWO", "COLTHREE", "COLFOUR"]
        my_rhs = [92.3, 14.8, 4]
        my_rownames = ["THISROW", "THATROW", "LASTROW"]
        my_rowsense = "GLG"

        model.objective.set_sense(model.objective.sense.minimize)
        model.objective.set_name("OBJECTIVE")
        model.variables.add(obj=my_obj, ub=my_ub, lb=my_lb, names=my_colnames)
        my_rows = [[[0, 1, 2, 3],
                    [0, 78.26, 0, 2.9]],
                   [["COLONE", "COLTWO", "COLTHREE", "COLFOUR"],
                    [0.24, 0, 11.31, 0]],
                   [[0, "COLTWO", "COLTHREE", 3],
                    [12.68, 0, 0.08, 0.9]]]
        model.linear_constraints.add(rhs=my_rhs,
                                     senses=my_rowsense,
                                     names=my_rownames,
                                     lin_expr=my_rows)
        #model.linear_constraints.set_coefficients("THISROW", 0, 9999)
        #model.objective.set_linear("COLONE", 9999)
        model.write("pytest1.lp")
        self.assertTrue(os.path.isfile("pytest1.lp"))
        self._failSafeDelete("pytest1.lp")
        model.solve()

        self.assertEqual(model.solution.get_status(),
                         model.solution.status.unbounded)
        self.assertAlmostEqual(model.solution.get_objective_value(),
                               6.23369085174,
                               places=6)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
