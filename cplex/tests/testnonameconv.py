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
Tests with CPLEX_PY_DISABLE_NAME_CONV=yes. That is, we disable name to
index conversion for performance.

No command line arguments are required.
"""
import importlib
import os
import unittest

import cplex  # KEEP: This is for NoNameConvTests.setUpClass.
from cplex import SparsePair
from cplextestcase import CplexTestCase


class NoNameConvTests(CplexTestCase):

    @classmethod
    def setUpClass(cls):
        # NOTE: We must set the CPLEX_PY_DISABLE_NAME_CONV environment
        #       variable *before* loading the _aux_functions module.
        os.environ["CPLEX_PY_DISABLE_NAME_CONV"] = "yes"
        importlib.reload(cplex._internal._aux_functions)

    @classmethod
    def tearDownClass(cls):
        del os.environ["CPLEX_PY_DISABLE_NAME_CONV"]
        importlib.reload(cplex._internal._aux_functions)

    def setUp(self):
        self.varnames = ["x{0}".format(x) for x in range(3)]
        self.cpx = self._newCplex()
        self.varind = list(self.cpx.variables.add(names=self.varnames))
        self.assertEqual(len(self.varnames), self.cpx.variables.get_num())
        self.nvars = len(self.varind)

    def tearDown(self):
        self.cpx.end()

    def testEnvVar(self):
        self.assertEqual(os.getenv("CPLEX_PY_DISABLE_NAME_CONV", None),
                         "yes")

    def testDeleteByIndex(self):
        self.cpx.variables.delete(self.varind)
        self.assertEqual(0, self.cpx.variables.get_num())

    def testDeleteByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.delete(self.varnames)

    def testGetLBByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.get_lower_bounds(self.varnames)

    def testSetLBByIndex(self):
        self.cpx.variables.set_lower_bounds([(i, 1.0) for i in self.varind])
        self.assertEqual([1.0] * self.nvars,
                         self.cpx.variables.get_lower_bounds())

    def testSetLBByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.set_lower_bounds([(n, 1.0)
                                                 for n in self.varnames])

    def testGetUBByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.get_upper_bounds(self.varnames)

    def testSetUBByIndex(self):
        self.cpx.variables.set_upper_bounds([(i, 1.0) for i in self.varind])
        self.assertEqual([1.0] * self.nvars,
                         self.cpx.variables.get_upper_bounds())

    def testSetUBByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.set_upper_bounds([(n, 1.0)
                                                 for n in self.varnames])

    def testSetNamesByIndex(self):
        self.cpx.variables.set_names([(i, n)
                                      for i, n
                                      in enumerate(self.varnames)])
        self.assertEqual(self.varnames, self.cpx.variables.get_names())

    def testSetNamesByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.set_names([(n, n) for n in self.varnames])

    def testSetTypesByIndex(self):
        self.cpx.variables.set_types([(i, 'C') for i in self.varind])
        self.assertEqual(['C'] * self.nvars,
                         self.cpx.variables.get_types())

    def testSetTypesByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.set_types([(n, 'C') for n in self.varnames])


    def testGetColsByIndex(self):
        cols = self.cpx.variables.get_cols(self.varind)
        self.assertEqual(self.nvars, len(cols))
        expected = [SparsePair(ind=[], val=[])] * self.nvars
        for e, c in zip(cols, expected):
            self.assertEqual(e.ind, c.ind)
            self.assertEqual(e.val, c.val)

    def testGetColsByName(self):
        with self.assertRaises(TypeError):
            self.cpx.variables.get_cols(self.varnames)

    def testSetQuadraticByIndex(self):
        self.cpx.objective.set_quadratic([[self.varind, [1.0] * self.nvars]
                                          for _ in range(self.nvars)])
        self.assertEqual(self.nvars ** 2,
                         self.cpx.objective.get_num_quadratic_nonzeros())

    def testSetQuadraticByName(self):
        with self.assertRaises(TypeError):
            self.cpx.objective.set_quadratic([[self.varnames,
                                               [1.0] * self.nvars]
                                              for _ in range(self.nvars)])

    def testAddLinConstrByName(self):
        # We use names here, but it is expected to work because the
        # name-to-index conversion is done in the SWIG C layer (i.e.,
        # it is not affected by CPLEX_PY_DISABLE_NAME_CONV).
        self.cpx.linear_constraints.add(
            lin_expr=[[self.varnames, [0.0] * self.nvars]],
            senses="L",
            rhs=[1.0])
        self.assertEqual(1, self.cpx.linear_constraints.get_num())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
