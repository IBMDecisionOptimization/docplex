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
Tests Quadratic Constraint API.

No command line arguments are required.
"""
import unittest
import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
from interfacetestcase import InterfaceTestCase, override


class QuadraticConstraintTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        return cpx.quadratic_constraints

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        names = self.getTestNames(cpx)
        indices = []
        iface = self.get_interface(cpx)
        var = cpx.variables
        self.assertEqual(iface.get_num(), 0)
        self.assertEqual(var.get_num(), 0)
        var.add(names=['a', 'b'])
        sp = cplex.SparsePair(ind=['a'], val=[1.0])
        st = cplex.SparseTriple(ind1=['a'], ind2=['b'], val=[1.0])
        for name in names:
            idx = iface.add(lin_expr=sp,
                            quad_expr=st,
                            sense='G',
                            rhs=1.0,
                            name=name)
            indices.append(idx)
        self.assertEqual(len(indices), len(names))
        self.assertEqual(iface.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        iface = self.get_interface(cpx)
        [idx1, idx2] = cpx.variables.add(lb=[0.0, 0.0])
        sp = cplex.SparsePair(ind=[idx1], val=[1.0])
        st = cplex.SparseTriple(ind1=[idx1], ind2=[idx2], val=[1.0])
        idx = iface.add(lin_expr=sp,
                        quad_expr=st,
                        sense='G',
                        rhs=1.0,
                        name=name)
        return idx

    @override(InterfaceTestCase)
    def defaultName(self):
        return "q"

    # FIXME?: This isn't bad necessarily, but for indicators we get a
    # CPXERR_NO_NAMES here rather than a CPXERR_NAME_NOT_FOUND.  Thus,
    # we override this test to avoid a failure.  We should probably make
    # this consistent one way or the other.
    @override(InterfaceTestCase)
    def testGetIndicesNoNames(self):
        """RTC-31974"""
        try:
            with self._newCplex() as cpx:
                iface = self.get_interface(cpx)
                idx = iface.get_indices("bogus")
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_NAMES)


class QuadraticConstraintEncodingTests(QuadraticConstraintTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
