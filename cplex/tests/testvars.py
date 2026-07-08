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
Tests Variable Interface API.

No command line arguments are required.
"""
import unittest
import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
from interfacetestcase import InterfaceTestCase, override


class VariablesInterfaceTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        return cpx.variables

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        names = self.getTestNames(cpx)
        iface = self.get_interface(cpx)
        self.assertEqual(iface.get_num(), 0)
        ctypes = self.getnumitems(list(cpx.variables.type), len(names))
        indices= list(iface.add(types=ctypes, names=names))
        self.assertEqual(len(indices), len(names))
        self.assertEqual(iface.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        iface = self.get_interface(cpx)
        [idx] = iface.add(lb=[0.0], names=[name])
        return idx

    @override(InterfaceTestCase)
    def defaultName(self):
        return "x"

    @unittest.skip("FIXME")
    @override(InterfaceTestCase)
    def testAddOneEmptyString(self):
        # We get an empty string back instead of a default name.
        InterfaceTestCase.testAddOneEmptyString(self)

    @unittest.skip("FIXME")
    @override(InterfaceTestCase)
    def testGetIndicesNoNames(self):
        # We get a CPXERR_NO_NAMES instead of a CPXERR_NAME_NOT_FOUND.
        InterfaceTestCase.testGetIndicesNoNames(self)


class VariablesInterfaceEncodingTests(VariablesInterfaceTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
