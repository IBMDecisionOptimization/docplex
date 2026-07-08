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
Tests SOS Constraint API.

No command line arguments are required.
"""
import unittest
import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
from interfacetestcase import InterfaceTestCase, override


class SOSConstraintTests(InterfaceTestCase, CplexTestCase):

    @override(InterfaceTestCase)
    def get_interface(self, cpx):
        return cpx.SOS

    @override(InterfaceTestCase)
    def doSetUp(self, cpx):
        names = self.getTestNames(cpx)
        indices = []
        iface = self.get_interface(cpx)
        var = cpx.variables
        self.assertEqual(iface.get_num(), 0)
        self.assertEqual(var.get_num(), 0)
        var.add(names=['a', 'b'])
        sp = cplex.SparsePair(ind=['a', 'b'], val=[1.0, 2.0])
        for name in names:
            idx = iface.add(type='1', SOS=sp, name=name)
            indices.append(idx)
        self.assertEqual(len(indices), len(names))
        self.assertEqual(iface.get_num(), len(indices))
        return names, indices

    @override(InterfaceTestCase)
    def addOne(self, cpx, name):
        iface = self.get_interface(cpx)
        [idx1, idx2] = cpx.variables.add(lb=[0.0, 0.0])
        sp = cplex.SparsePair(ind=[idx1, idx2], val=[1.0, 2.0])
        # FIXME: we should not use 'type' as an argument name; it is the
        # name of a built-in function.  Replace with "sostype".
        idx = iface.add(type='1', SOS=sp, name=name)
        return idx

    @override(InterfaceTestCase)
    def defaultName(self):
        return "s"

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

    def testSolveAfterSOSDelete(self):
        """RTC-32217"""
        with self._newCplex() as cpx:
            size = 7
            allvars = cpx.variables.add(ub=[10] * size)
            cpx.SOS.add(type=cpx.SOS.type.SOS2,
                        SOS=cplex.SparsePair(ind=allvars,
                                             val=range(1, size + 1)))
            # maximize sum
            cpx.objective.set_linear([(idx, 1) for idx in allvars])
            cpx.objective.set_sense(cpx.objective.sense.maximize)
            cpx.solve()
            # result should be 2 * 10 = 20
            self.assertEqual(20, cpx.solution.get_objective_value())
            # remove all SOS and solve again
            cpx.SOS.delete()
            self.assertEqual(0, cpx.SOS.get_num())
            cpx.solve()
            # result should be 7 * 10 = 70
            self.assertEqual(70, cpx.solution.get_objective_value())


class SOSConstraintEncodingTests(SOSConstraintTests):
    # Do some basic tests with API encoding parameter.
    test_encoding = True


def main():
    unittest.main()

if __name__ == '__main__':
    main()
