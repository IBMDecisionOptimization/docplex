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
Tests the MIPStartsInterface.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase


class MipStartsInterfaceTests(CplexTestCase):

    def testIteratorFromAdd(self):
        cpx = self._newCplex()
        varind = cpx.variables.add(names=['a', 'b', 'c'], types="III")
        indices = cpx.MIP_starts.add(
            [(cplex.SparsePair(ind=[i], val=[0.0]),
              cpx.MIP_starts.effort_level.auto)
             for i in varind])
        self.assertEqual([0, 1, 2], list(indices))
        varind = cpx.variables.add(names=['d', 'e', 'f'], types="III")
        indices = cpx.MIP_starts.add(
            [(cplex.SparsePair(ind=[i], val=[0.0]),
              cpx.MIP_starts.effort_level.auto)
             for i in varind])
        self.assertEqual([3, 4, 5], list(indices))

    def testIteratorFromAddAfterDelete(self):
        cpx = self._newCplex()
        varind = cpx.variables.add(types="III")
        indices = cpx.MIP_starts.add(
            [(cplex.SparsePair(ind=[i], val=[0.0]),
              cpx.MIP_starts.effort_level.auto,
              'ms{0}'.format(i))
             for i in varind])
        self.assertEqual([0, 1, 2], list(indices))
        self.assertEqual('ms0', cpx.MIP_starts.get_names(0))
        self.assertEqual('ms1', cpx.MIP_starts.get_names(1))
        self.assertEqual('ms2', cpx.MIP_starts.get_names(2))
        cpx.MIP_starts.delete(1)
        self.assertEqual(2, cpx.MIP_starts.get_num())
        # NB: The index of 'ms2' has changed.  This is the expected behavior.
        self.assertEqual('ms2', cpx.MIP_starts.get_names(1))
        indices = cpx.MIP_starts.add(
            cplex.SparsePair(ind=[0], val=[1.0]),
            cpx.MIP_starts.effort_level.auto,
            'ms3')
        self.assertEqual([2], list(indices))
        self.assertEqual('ms0', cpx.MIP_starts.get_names(0))
        self.assertEqual('ms2', cpx.MIP_starts.get_names(1))
        self.assertEqual('ms3', cpx.MIP_starts.get_names(2))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
