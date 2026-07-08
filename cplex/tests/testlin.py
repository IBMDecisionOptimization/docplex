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
Tests Linear Constraint API.

No command line arguments are required.
"""
import unittest
import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
from interfacetestcase import InterfaceTestCase, override


class LinearConstrTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        return cpx.linear_constraints

    # FIXME: We should be able to iterate these
    @staticmethod
    def getsenses():
        return ['G', 'L', 'E', 'R']

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        names = self.getTestNames(cpx)
        iface = self.get_interface(cpx)
        self.assertEqual(iface.get_num(), 0)
        var = cpx.variables
        self.assertEqual(var.get_num(), 0)
        var.add(names=['a', 'b'])
        sp = cplex.SparsePair(ind=['a', 'b'], val=[1.0, 2.0])
        lin_expr = [sp] * len(names)
        senses = self.getnumitems(self.getsenses(), len(names))
        indices= list(iface.add(lin_expr=lin_expr, names=names))
        self.assertEqual(len(indices), len(names))
        self.assertEqual(iface.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        iface = self.get_interface(cpx)
        [varidx] = cpx.variables.add(lb=[0.0])
        sp = cplex.SparsePair(ind=[varidx], val=[1.0])
        [idx] = iface.add(lin_expr=[sp], names=[name])
        return idx

    @override(InterfaceTestCase)
    def defaultName(self):
        return "c"

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


class LinearConstrEncodingTests(LinearConstrTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
