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
Tests Solution Pool API.

No command line arguments are required.
"""
import os
import unittest
from cplex.exceptions import CplexError, CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
from interfacetestcase import InterfaceTestCase, override


class SolutionPoolTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        return cpx.solution.pool

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        cpx.read("../../data/location_lin.lp")
        # In parallel, populate may not respect this parameter exactly
        # due to disparities between threads.  That is, it may happen
        # that populate stops when it has generated a number of solutions
        # slightly more than or slightly less than this limit because of
        # differences in synchronization between threads.
        cpx.parameters.mip.limits.populate.set(5)
        cpx.populate_solution_pool()
        iface = self.get_interface(cpx)
        # We only check that there is at least one solution in the pool
        # (see comment above).
        self.assertGreater(iface.get_num(), 0)
        names = iface.get_names()
        indices = iface.get_indices(names)
        return names, indices

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testAddOne(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testAddOneNoNames(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testAddOneNone(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testAddOneEmptyString(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testDeleteIntermixed(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testGetIndicesNoNames(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testGetNameAllNoNames(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testGetNameNoNames(self):
        pass

    @unittest.skip("Not relevant for solution pools")
    @override(InterfaceTestCase)
    def testGetNameRangeNoNames(self):
        pass

    @override(InterfaceTestCase)
    def testWithMultipleInstances(self):
        """RTC-33155"""
        cpx1 = self._newCplex()
        iface1 = self.get_interface(cpx1)
        self.doSetUp(cpx1)
        self.assertGreater(iface1.get_num(), 0)
        cpx2 = self._newCplex()
        iface2 = self.get_interface(cpx2)
        self.assertEqual(iface2.get_num(), 0)
        self.assertGreater(iface1.get_num(), 0)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
