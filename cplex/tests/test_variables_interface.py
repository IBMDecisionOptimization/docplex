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
Tests the VariablesInterface.

No command line arguments are required.
"""
import unittest
from cplex.exceptions import CplexSolverError
from cplextestcase import CplexTestCase
from cplex.exceptions import error_codes
from cplex._internal._constants import (CPX_CONTINUOUS, CPX_BINARY,
                                        CPX_INTEGER, CPX_SEMICONT,
                                        CPX_SEMIINT)


class VariablesTests(CplexTestCase):

    def testIteratorFromAddEmpty(self):
        cpx = self._newCplex()
        indices = cpx.variables.add()
        self.assertEqual([], list(indices))
        self.assertEqual(0, cpx.variables.get_num())

    def testIteratorFromAdd(self):
        cpx = self._newCplex()
        indices = cpx.variables.add(names=['a', 'b', 'c'])
        self.assertEqual([0, 1, 2], list(indices))
        indices = cpx.variables.add(names=['d', 'e', 'f'])
        self.assertEqual([3, 4, 5], list(indices))

    def testIteratorFromAddAfterDelete(self):
        cpx = self._newCplex()
        indices = cpx.variables.add(names=['a', 'b', 'c'])
        self.assertEqual([0, 1, 2], list(indices))
        self.assertEqual('c', cpx.variables.get_names(2))
        cpx.variables.delete(1)
        self.assertEqual(2, cpx.variables.get_num())
        # NB: The index of 'c' has changed.  This is the expected behavior.
        self.assertEqual('c', cpx.variables.get_names(1))
        indices = cpx.variables.add(names=['d'])
        self.assertEqual([2], list(indices))
        self.assertEqual('a', cpx.variables.get_names(0))
        self.assertEqual('c', cpx.variables.get_names(1))
        self.assertEqual('d', cpx.variables.get_names(2))

    def testIteratorFromAddInLoop(self):
        cpx = self._newCplex()
        # NB: This is a potential work-around for RTC-23753.
        for idx in cpx.variables.add(lb=[0, 0, 0]):
            cpx.variables.set_names(idx, 'x{0}'.format(idx))
        self.assertEqual(['x0', 'x1', 'x2'], cpx.variables.get_names())

    def testGetTypesSingle(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'], types=[cpx.variables.type.continuous])
        # test with single string
        self.assertEqual(cpx.variables.get_types('x'),
                         cpx.variables.type.continuous)
        # test with single index
        self.assertEqual(cpx.variables.get_types(0),
                         cpx.variables.type.continuous)
        # test with range of indices
        self.assertEqual(cpx.variables.get_types(0, 0),
                         cpx.variables.type.continuous)
        # test with list of strings
        self.assertEqual(cpx.variables.get_types(['x']),
                         cpx.variables.type.continuous)
        # test with list of indices
        self.assertEqual(cpx.variables.get_types([0]),
                         cpx.variables.type.continuous)
        # test with no args
        self.assertEqual(cpx.variables.get_types(),
                         cpx.variables.type.continuous)

    def testGetTypesMultiple(self):
        cpx = self._newCplex()
        varlst = [vt for vt in cpx.variables.type]
        varlstlen = len(varlst)
        self.assertEqual(varlstlen, 5,
                         "Need to update test if this changes.")
        cpx.variables.add(names=varlst, types=varlst)
        # test with single string
        self.assertEqual(cpx.variables.get_types(varlst[0]), varlst[0])
        # test with single index
        self.assertEqual(cpx.variables.get_types(0), varlst[0])
        # test with range of indices
        self.assertEqual(cpx.variables.get_types(0, varlstlen - 1), varlst)
        # test with list of strings
        self.assertEqual(cpx.variables.get_types(varlst), varlst)
        # test with list of indices
        self.assertEqual(
            cpx.variables.get_types(list(range(varlstlen))),
            varlst)
        # test with no args
        self.assertEqual(cpx.variables.get_types(), varlst)
        # TODO: It would be nice if we could pass an iterator too.
        # For example, rather than having to wrap range with list above
        # we could just pass range.

    def testSetTypesSingle(self):
        cpx = self._newCplex()
        # By setting the types to continous, this should effectively
        # also make the problem a MIP
        cpx.variables.add(types=['C' for _ in cpx.variables.type])
        for idx, key in enumerate(cpx.variables.type):
            self.assertEqual(cpx.variables.get_types(idx), 'C')
            cpx.variables.set_types(idx, key)
            self.assertEqual(cpx.variables.get_types(idx), key)

    def testSetTypesMultiple(self):
        cpx = self._newCplex()
        cpx.variables.add(types=['C' for _ in cpx.variables.type])
        cpx.variables.set_types([(idx, key) for idx, key
                                 in enumerate(cpx.variables.type)])
        for idx, key in enumerate(cpx.variables.type):
            self.assertEqual(cpx.variables.get_types(idx), key)

    def testGetTypeName(self):
        cpx = self._newCplex()
        for vt in cpx.variables.type:
            vtname = cpx.variables.type[vt]
            self.assertTrue(len(vtname) > 0)

    def testGetTypeNameBad(self):
        cpx = self._newCplex()
        with self.assertRaises(KeyError):
            vtname = cpx.variables.type['bogus']

    def testAddNameOneNone(self):
        try:
            with self._newCplex() as cpx:
                indices = cpx.variables.add(names=[None])
                cpx.variables.get_names()
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NO_NAMES)

    def testAddNameTwoNones(self):
        """RTC-31860"""
        try:
            with self._newCplex() as cpx:
                indices = cpx.variables.add(names=[None, None])
                cpx.variables.get_names()
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NO_NAMES)

    def testGetNameNoneAugoGen(self):
        with self._newCplex() as cpx:
            indices = cpx.variables.add(names=['a', None])
            self.assertEqual(cpx.variables.get_names(0), 'a')
            # A name is auto-generated, which seems reasonable.
            self.assertEqual(cpx.variables.get_names(1), 'x2')

    def testSetLowerBoundWithGenerator(self):
        cpx = self._newCplex()
        indices = cpx.variables.add(lb=[0] * 3)
        cpx.variables.set_lower_bounds((i, 1) for i in indices)
        self.assertEqual(cpx.variables.get_lower_bounds(), [1] * 3)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
