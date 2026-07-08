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
Tests the SOSInterface.

No command line arguments are required.
"""
import unittest
from cplex.exceptions import CplexSolverError
from cplextestcase import CplexTestCase


class SosInterfaceTests(CplexTestCase):

    def testAddWithoutVariable(self):
        cpx = self._newCplex()
        try:
            cpx.SOS.add()
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], 1201)

    def testIndexFromAddEmpty(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'])
        idx = cpx.SOS.add()
        self.assertEqual(0, idx)
        self.assertEqual(1, cpx.SOS.get_num())

    def testIndexFromAdd(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'])
        indices = []
        for i in 'abc':
            indices.append(cpx.SOS.add(name=i))
        self.assertEqual([0, 1, 2], indices)
        for i in 'def':
            indices.append(cpx.SOS.add(name=i))
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)
        self.assertEqual(['a', 'b', 'c', 'd', 'e', 'f'],
                         cpx.SOS.get_names())

    def testIndexFromAddAfterDelete(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'])
        indices = []
        for i in 'abc':
            indices.append(cpx.SOS.add(name=i))
        self.assertEqual([0, 1, 2], indices)
        self.assertEqual('c', cpx.SOS.get_names(2))
        cpx.SOS.delete(1)
        self.assertEqual(2, cpx.SOS.get_num())
        # NB: The index of 'c' has changed.  This is the expected behavior.
        self.assertEqual('c', cpx.SOS.get_names(1))
        idx = cpx.SOS.add(name='d')
        self.assertEqual(2, idx)
        self.assertEqual('a', cpx.SOS.get_names(0))
        self.assertEqual('c', cpx.SOS.get_names(1))
        self.assertEqual('d', cpx.SOS.get_names(2))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
