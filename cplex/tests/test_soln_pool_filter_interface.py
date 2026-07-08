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
Tests the SolnPoolFilterInterface.

No command line arguments are required.
"""
import unittest
import os

from cplex.exceptions import CplexError, CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase


class SolnPoolFilterInterfaceTests(CplexTestCase):

    # TODO: test the rest of the methods

    def testAddDivWithBadVariableType(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'], types='C')
        try:
            cpx.solution.pool.filter.add_diversity_filter(
                0, 10, [[0], [1]], [1], '')
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_FILTER_VARIABLE_TYPE)

    def testAddDivWithInconsistentArgs(self):
        if not __debug__:
            print("Skipping testAddDivWithInconsistentArgs when not __debug__!")
            return
        # weights with different len than expression
        self.addDivWithInconsistentArgs(0, 10, [[0], [1]], [1, 2], '')
        # expression with different arg len
        self.addDivWithInconsistentArgs(0, 10, [[0, 1], [1]], [1], '')

    def addDivWithInconsistentArgs(self, lb, ub, exp, wghts, name):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'], types='C')
        try:
            cpx.solution.pool.filter.add_diversity_filter(
                lb, ub, exp, wghts, name)
            self.fail()
        except CplexError as err:
            self.assertIn('inconsistent argument lengths', str(err))

    def testDivWithBadArgsOptimized(self):
        if __debug__:
            print("Skipping testDivWithBadArgsOptimized when __debug__!")
            return
        cpx = self._newCplex()
        cpx.variables.add(types='BB')
        try:
            cpx.solution.pool.filter.add_diversity_filter(
                0, 10, [[0, 1], []], [1], '')
            self.fail()
        except CplexSolverError as err:
            # When running with non-optimized bytecode, we would never
            # get this error. We'd get an "inconsistent argument lengths"
            # error instead.
            self.assertEqual(err.args[2], error_codes.CPXERR_NULL_POINTER)

    def testAddRngWithBadVariableType(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'])
        try:
            cpx.solution.pool.filter.add_range_filter(
                0, 10, [[0], [1]], '')
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2],
                             error_codes.CPXERR_NOT_MIP)

    def testIndexFromAddDiv(self):
        cpx = self._newCplex()
        [varidx] = list(cpx.variables.add(names=['x'], types='B'))
        indices = []
        for fname in 'abc':
            indices.append(cpx.solution.pool.filter.add_diversity_filter(
                lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
                weights=[1.0], name=fname))
        self.assertEqual([0, 1, 2], indices)
        for fname in 'def':
            indices.append(cpx.solution.pool.filter.add_diversity_filter(
                lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
                weights=[1.0], name=fname))
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)
        self.assertEqual(['a', 'b', 'c', 'd', 'e', 'f'],
                         cpx.solution.pool.filter.get_names())

    def testIndexFromAddDivWithDefaultName(self):
        cpx = self._newCplex()
        [varidx] = list(cpx.variables.add(names=['x'], types='B'))
        # try with default name
        dfidx = cpx.solution.pool.filter.add_diversity_filter(
            lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
            weights=[2.0])
        self.assertEqual(0, dfidx)
        # if we use an empty name then a default name is given
        self.assertEqual('f1', cpx.solution.pool.filter.get_names(dfidx))
        # sanity check with getter
        (sp, wghts) = cpx.solution.pool.filter.get_diversity_filters(dfidx)
        self.assertEqual(([varidx], [1.0]), sp.unpack())
        self.assertEqual([2.0], wghts)

    def testIndexFromAddDivWithNameNone(self):
        cpx = self._newCplex()
        [varidx] = list(cpx.variables.add(names=['x'], types='B'))
        # try with name=None
        dfidx = cpx.solution.pool.filter.add_diversity_filter(
            lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
            weights=[2.0], name=None)
        self.assertEqual(0, dfidx)
        # if name is None , then we should get CPXERR_NO_NAMES
        try:
            cpx.solution.pool.filter.get_names(dfidx)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_NAMES)
        # sanity check with getter
        (sp, wghts) = cpx.solution.pool.filter.get_diversity_filters(dfidx)
        self.assertEqual(([varidx], [1.0]), sp.unpack())
        self.assertEqual([2.0], wghts)

    def testIndexFromAddDivWithDefaultWeights(self):
        cpx = self._newCplex()
        [varidx] = list(cpx.variables.add(names=['x'], types='B'))
        # try with default weights
        dfidx = cpx.solution.pool.filter.add_diversity_filter(
            lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
            name='')
        self.assertEqual(0, dfidx)
        # if we use an empty name then a default name is given
        self.assertEqual('f1', cpx.solution.pool.filter.get_names(dfidx))
        # sanity check with getter
        (sp, wghts) = cpx.solution.pool.filter.get_diversity_filters(dfidx)
        self.assertEqual(([varidx], [1.0]), sp.unpack())
        # When weights=[] then 1.0's should be used.
        self.assertEqual([1.0], wghts)

    def testIndexFromAddRng(self):
        cpx = self._newCplex()
        [varidx] = list(cpx.variables.add(names=['x'], types='I'))
        indices = []
        for fname in 'abc':
            indices.append(cpx.solution.pool.filter.add_range_filter(
                lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
                name=fname))
        self.assertEqual([0, 1, 2], indices)
        for fname in 'def':
            indices.append(cpx.solution.pool.filter.add_range_filter(
                lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
                name=fname))
        self.assertEqual([0, 1, 2, 3, 4, 5], indices)
        self.assertEqual(['a', 'b', 'c', 'd', 'e', 'f'],
                         cpx.solution.pool.filter.get_names())

    def testIndexFromAddRngWithDefaultName(self):
        cpx = self._newCplex()
        [varidx] = list(cpx.variables.add(names=['x'], types='I'))
        rngidx = cpx.solution.pool.filter.add_range_filter(
            lb=0.0, ub=10.0, expression=[[varidx], [1.0]])
        # if we use an empty name then a default name is given
        self.assertEqual('f1', cpx.solution.pool.filter.get_names(rngidx))
        # sanity check with getter
        sp = cpx.solution.pool.filter.get_range_filters(rngidx)
        self.assertEqual(([varidx], [1.0]), sp.unpack())

    def testIndexFromAddRngWithNameNone(self):
        cpx = self._newCplex()
        [varidx] = list(cpx.variables.add(names=['x'], types='I'))
        rngidx = cpx.solution.pool.filter.add_range_filter(
            lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
            name=None)
        # if name=None, then we should get CPXERR_NO_NAMES
        try:
            cpx.solution.pool.filter.get_names(rngidx)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_NAMES)
        # sanity check with getter
        sp = cpx.solution.pool.filter.get_range_filters(rngidx)
        self.assertEqual(([varidx], [1.0]), sp.unpack())

    def testIndexFromAddDivAfterDelete(self):
        cpx = self._newCplex()
        varindices = cpx.variables.add(names=['x'], types='B')
        self.assertEqual(0, list(varindices)[0])
        varidx = 0
        indices = []
        for fname in 'abc':
            indices.append(cpx.solution.pool.filter.add_diversity_filter(
                lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
                weights=[1.0], name=fname))
        self.assertEqual([0, 1, 2], indices)
        self.assertEqual('c', cpx.solution.pool.filter.get_names(2))
        cpx.solution.pool.filter.delete(1)
        self.assertEqual(2, cpx.solution.pool.filter.get_num())
        # NB: The index of 'c' has changed.  This is the expected behavior.
        self.assertEqual('c', cpx.solution.pool.filter.get_names(1))
        idx = cpx.solution.pool.filter.add_diversity_filter(
            lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
            weights=[1.0], name='d')
        self.assertEqual(2, idx)
        self.assertEqual('a', cpx.solution.pool.filter.get_names(0))
        self.assertEqual('c', cpx.solution.pool.filter.get_names(1))
        self.assertEqual('d', cpx.solution.pool.filter.get_names(2))

    def testIndexFromAddRngAfterDelete(self):
        cpx = self._newCplex()
        varindices = cpx.variables.add(names=['x'], types='I')
        self.assertEqual(0, list(varindices)[0])
        varidx = 0
        indices = []
        for fname in 'abc':
            indices.append(cpx.solution.pool.filter.add_range_filter(
                lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
                name=fname))
        self.assertEqual([0, 1, 2], indices)
        self.assertEqual('c', cpx.solution.pool.filter.get_names(2))
        cpx.solution.pool.filter.delete(1)
        self.assertEqual(2, cpx.solution.pool.filter.get_num())
        # NB: The index of 'c' has changed.  This is the expected behavior.
        self.assertEqual('c', cpx.solution.pool.filter.get_names(1))
        idx = cpx.solution.pool.filter.add_range_filter(
            lb=0.0, ub=10.0, expression=[[varidx], [1.0]],
            name='d')
        self.assertEqual(2, idx)
        self.assertEqual('a', cpx.solution.pool.filter.get_names(0))
        self.assertEqual('c', cpx.solution.pool.filter.get_names(1))
        self.assertEqual('d', cpx.solution.pool.filter.get_names(2))

    def testIndexFromAddBoth(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['x'], types='B')
        dividx = cpx.solution.pool.filter.add_diversity_filter(
            lb=0.0, ub=10.0, expression=[[0], [1.0]],
            weights=[1.0], name='a')
        rngidx = cpx.solution.pool.filter.add_range_filter(
            lb=0.0, ub=10.0, expression=[[0], [1.0]], name='b')
        self.assertEqual(['a', 'b'], cpx.solution.pool.filter.get_names())
        self.assertEqual(cpx.solution.pool.filter.type.diversity,
                         cpx.solution.pool.filter.get_types(dividx))
        self.assertEqual(cpx.solution.pool.filter.type.diversity,
                         cpx.solution.pool.filter.get_types('a'))
        self.assertEqual(cpx.solution.pool.filter.type.range,
                         cpx.solution.pool.filter.get_types(rngidx))
        self.assertEqual(cpx.solution.pool.filter.type.range,
                         cpx.solution.pool.filter.get_types('b'))

    def testReadWrite(self):
        cpx = self._newCplex()
        cpxReader = self._newCplex()
        cpx.variables.add(names=['x'], types='B')
        with self._getTempFileName() as tmp:
            cpx.write(tmp, filetype='sav')
            cpxReader.read(tmp, filetype='sav')
        dividx = cpx.solution.pool.filter.add_diversity_filter(
            lb=0.0, ub=10.0, expression=[[0], [1.0]],
            weights=[1.0], name='a')
        rngidx = cpx.solution.pool.filter.add_range_filter(
            lb=0.0, ub=10.0, expression=[[0], [1.0]], name='b')
        with self._getTempFileName() as tmp:
            cpx.solution.pool.filter.write(tmp)
            self.assertTrue(os.path.exists(tmp))
            cpxReader.solution.pool.filter.read(tmp)
        self.assertEqual(2, cpx.solution.pool.filter.get_num())
        self.assertEqual(2, cpxReader.solution.pool.filter.get_num())


def main():
    unittest.main()

if __name__ == '__main__':
    main()
