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
Tests the IndicatorConstraintInterface.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase
from cplex import SparsePair
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes

VAR_0_NAME = 'x1'
VAR_1_NAME = 'x2'
IND_NAME = 'ind1'
IND_RHS = 1.0
IND_SENSE = 'G'
IND_VAL = 2.0


class IndicatorTests(CplexTestCase):

    # FIXME: Should be True/False not 0/1
    complemented = 0

    def testIndexFromAddEmpty(self):
        cpx = self._newCplex()
        idx = cpx.indicator_constraints.add()
        self.assertEqual(0, idx)
        self.assertEqual(1, cpx.indicator_constraints.get_num())

    def testIndexFromAdd(self):
        cpx = self._newCplex()
        indices = []
        for i in 'abc':
            indices.append(cpx.indicator_constraints.add(name=i))
        self.assertEqual([0, 1, 2], indices)
        for i in 'def':
            indices.append(cpx.indicator_constraints.add(name=i))
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)
        self.assertEqual(['a', 'b', 'c', 'd', 'e', 'f'],
                         cpx.indicator_constraints.get_names())

    def testIndexFromAddAfterDelete(self):
        cpx = self._newCplex()
        indices = []
        for i in 'abc':
            indices.append(cpx.indicator_constraints.add(name=i))
        self.assertEqual([0, 1, 2], indices)
        self.assertEqual('c', cpx.indicator_constraints.get_names(2))
        cpx.indicator_constraints.delete(1)
        self.assertEqual(2, cpx.indicator_constraints.get_num())
        # NB: The index of 'c' has changed.  This is the expected behavior.
        self.assertEqual('c', cpx.indicator_constraints.get_names(1))
        idx = cpx.indicator_constraints.add(name='d')
        self.assertEqual(2, idx)
        self.assertEqual('a', cpx.indicator_constraints.get_names(0))
        self.assertEqual('c', cpx.indicator_constraints.get_names(1))
        self.assertEqual('d', cpx.indicator_constraints.get_names(2))

    def testGetNumOnEmpty(self):
        cpx = self._newCplex()
        self.assertEqual(cpx.indicator_constraints.get_num(), 0)

    def testGetNamesOnEmtpy(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.get_names
        self._checkFourFormMethodOnEmpty(func)

    def testGetSensesOnEmpty(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.get_senses
        self._checkFourFormMethodOnEmpty(func)

    def testDeleteOnEmpty(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.delete
        self.assertEqual(func(), None)
        self.assertEqual(cpx.indicator_constraints.get_num(), 0)
        self._checkIndexException(func, 0)
        self._checkIndexException(func, [0])
        self._checkIndexException(func, 0, 0)

    def testGetRhsOnEmpty(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.get_rhs
        self._checkFourFormMethodOnEmpty(func)

    def testGetComplementedOnEmpty(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.get_complemented
        self._checkFourFormMethodOnEmpty(func)

    def testGetIndicatorVariablesOnEmpty(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.get_indicator_variables
        self._checkFourFormMethodOnEmpty(func)

    def testGetNumNonZerosOnEmpty(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.get_num_nonzeros
        self._checkFourFormMethodOnEmpty(func)

    def testGetLinearComponentsOnEmpty(self):
        cpx = self._newCplex()
        func = cpx.indicator_constraints.get_linear_components
        self._checkFourFormMethodOnEmpty(func)

    def testGetNumOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        self.assertEqual(cpx.indicator_constraints.get_num(), 1)

    def testGetNamesOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        func = cpx.indicator_constraints.get_names
        self._checkFourFormMethodOnSimple(func, IND_NAME)

    def testGetSensesOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        func = cpx.indicator_constraints.get_senses
        self._checkFourFormMethodOnSensesSimple(func, IND_SENSE)

    def testDeleteOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        cpx.indicator_constraints.delete() # no arg
        self.assertEqual(cpx.indicator_constraints.get_num(), 0)
        cpx = self._createSimpleIndicatorModel()
        cpx.indicator_constraints.delete(0) # index arg
        self.assertEqual(cpx.indicator_constraints.get_num(), 0)
        cpx = self._createSimpleIndicatorModel()
        cpx.indicator_constraints.delete([0]) # list arg
        self.assertEqual(cpx.indicator_constraints.get_num(), 0)
        cpx = self._createSimpleIndicatorModel()
        cpx.indicator_constraints.delete(0, 0) # begin/end arg
        self.assertEqual(cpx.indicator_constraints.get_num(), 0)

    def testGetRhsOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        func = cpx.indicator_constraints.get_rhs
        self._checkFourFormMethodOnSimple(func, IND_RHS)

    def testGetComplementedOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        func = cpx.indicator_constraints.get_complemented
        self._checkFourFormMethodOnSimple(func, self.complemented)

    def testGetIndicatorVariablesOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        func = cpx.indicator_constraints.get_indicator_variables
        self._checkFourFormMethodOnSimple(func, 0)

    def testGetNumNonZerosOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        # TODO: Check documentation on get_num_nonzeros,
        #       the example seemed wrong to me.
        func = cpx.indicator_constraints.get_num_nonzeros
        self._checkFourFormMethodOnSimple(func, 1)

    def testGetLinearComponentsOnSimple(self):
        cpx = self._createSimpleIndicatorModel()
        func = cpx.indicator_constraints.get_linear_components
        expected = SparsePair(ind=[1], val=[IND_VAL])
        assertStrEqual = lambda a, b: self.assertEqual(str(a), str(b))
        assertStrEqual(func(), [expected])
        assertStrEqual(func(0), expected)
        assertStrEqual(func([0]), [expected])
        assertStrEqual(func(0, 0), [expected])

    def _checkFourFormMethodOnSimple(self, func, expected):
        self.assertEqual(func(), [expected])
        self.assertEqual(func(0), expected)
        self.assertEqual(func([0]), [expected])
        self.assertEqual(func(0, 0), [expected])

    def _checkFourFormMethodOnSensesSimple(self, func, expected):
        self.assertEqual(func(), expected)
        self.assertEqual(func(0), expected)
        self.assertEqual(func([0]), expected)
        self.assertEqual(func(0, 0), expected)

    def _checkFourFormMethodOnEmpty(self, func):
        self.assertEqual(func(), [])
        self._checkIndexException(func, 0)
        self._checkIndexException(func, [0])
        self._checkIndexException(func, 0, 0)

    def _checkIndexException(self, func, *args):
        try:
            if len(args) == 1:
                func(args[0])
            elif len(args) == 2:
                func(args[0], args[1])
            else:
                self.fail('Unexpected argument!')
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_INDEX_RANGE)

    def _createSimpleIndicatorModel(self):
        cpx = self._newCplex()
        cpx.variables.add(names=[VAR_0_NAME, VAR_1_NAME])
        cpx.indicator_constraints.add(indvar=VAR_0_NAME,
                                      complemented=self.complemented,
                                      rhs=IND_RHS,
                                      sense=IND_SENSE,
                                      lin_expr=SparsePair(ind=[VAR_1_NAME],
                                                          val=[IND_VAL]),
                                      name=IND_NAME)
        return cpx


class IndicatorListTests(IndicatorTests):
    """Check that lin_expr argument works with a list of two lists too."""

    def _createSimpleIndicatorModel(self):
        cpx = self._newCplex()
        cpx.variables.add(names=[VAR_0_NAME, VAR_1_NAME])
        cpx.indicator_constraints.add(indvar=VAR_0_NAME,
                                      complemented=self.complemented,
                                      rhs=IND_RHS,
                                      sense=IND_SENSE,
                                      lin_expr=[[VAR_1_NAME], [IND_VAL]],
                                      name=IND_NAME)
        return cpx

class IndicatorComplementedTests(IndicatorTests):
    """Check that works with complemented too."""

    complemented = 1


def main():
    unittest.main()

if __name__ == '__main__':
    main()
