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
Tests MIP Starts API.

No command line arguments are required.
"""
import unittest
import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
from interfacetestcase import InterfaceTestCase, override


class MIPStartsTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        # FIXME: The "MIP" in "MIP_starts" should not be capitalized.
        return cpx.MIP_starts

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        names = self.getTestNames(cpx)
        iface = self.get_interface(cpx)
        var = cpx.variables
        self.assertEqual(iface.get_num(), 0)
        self.assertEqual(var.get_num(), 0)
        var.add(names=['a', 'b'], types="II")
        sp = cplex.SparsePair(ind=['a', 'b'], val=[1.0, 2.0])
        indices= list(iface.add([(sp, effortlevel, name)
                                 for effortlevel, name in
                                 zip(cpx.MIP_starts.effort_level, names)]))
        self.assertEqual(len(indices), len(names))
        self.assertEqual(iface.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        iface = self.get_interface(cpx)
        [idx1, idx2] = cpx.variables.add(lb=[0.0, 0.0], types="II")
        sp = cplex.SparsePair(ind=[idx1, idx2], val=[1.0, 2.0])
        [idx] = iface.add(sp, cpx.MIP_starts.effort_level.auto, name)
        return idx

    @override(InterfaceTestCase)
    def defaultName(self):
        return "m"

    # FIXME: This seems like a bug ... we get a CPXERR_NOT_MIP here
    # rather than a CPXERR_NAME_NOT_FOUND.  At the least, this is quite
    # different than what we get with the other interfaces.
    @override(InterfaceTestCase)
    def testGetIndicesNoNames(self):
        """RTC-31974"""
        try:
            with self._newCplex() as cpx:
                iface = self.get_interface(cpx)
                idx = iface.get_indices("bogus")
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_MIP)


class MIPStartsEncodingTests(MIPStartsTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
