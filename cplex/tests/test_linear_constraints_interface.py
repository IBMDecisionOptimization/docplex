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
Tests the LinearConstraintsInterface.

No command line arguments are required.
"""
from collections import namedtuple
import unittest

from cplex import SparsePair
from cplex.exceptions import WrongNumberOfArgumentsError
from cplextestcase import CplexTestCase

ZeroMatrix = namedtuple("ZeroMatrix", ["model",
                                       "varind", "varnames",
                                       "conind", "connames"])
sensetypes = ['L', 'G', 'E', 'R']


class LinearConstraintsTests(CplexTestCase):

    def testIteratorFromAddEmpty(self):
        cpx = self._newCplex()
        indices = cpx.linear_constraints.add()
        self.assertEqual([], list(indices))
        self.assertEqual(0, cpx.linear_constraints.get_num())

    def testIteratorFromAdd(self):
        cpx = self._newCplex()
        indices = cpx.linear_constraints.add(names=['a', 'b', 'c'])
        self.assertEqual([0, 1, 2], list(indices))
        indices = cpx.linear_constraints.add(names=['d', 'e', 'f'])
        self.assertEqual([3, 4, 5], list(indices))

    def testIteratorFromAddAfterDelete(self):
        cpx = self._newCplex()
        indices = cpx.linear_constraints.add(names=['a', 'b', 'c'])
        self.assertEqual([0, 1, 2], list(indices))
        self.assertEqual('c', cpx.linear_constraints.get_names(2))
        cpx.linear_constraints.delete(1)
        self.assertEqual(2, cpx.linear_constraints.get_num())
        # NB: The index of 'c' has changed.  This is the expected behavior.
        self.assertEqual('c', cpx.linear_constraints.get_names(1))
        indices = cpx.linear_constraints.add(names=['d'])
        self.assertEqual([2], list(indices))
        self.assertEqual('a', cpx.linear_constraints.get_names(0))
        self.assertEqual('c', cpx.linear_constraints.get_names(1))
        self.assertEqual('d', cpx.linear_constraints.get_names(2))

    def testIteratorFromAddInLoop(self):
        cpx = self._newCplex()
        for idx in cpx.linear_constraints.add(rhs=[0, 0, 0]):
            cpx.linear_constraints.set_names(idx, 'x{0}'.format(idx))
        self.assertEqual(['x0', 'x1', 'x2'],
                         cpx.linear_constraints.get_names())

    def testGetRangeValuesNoArg(self):
        cpx = self._newCplex()
        rnglst = cpx.linear_constraints.get_range_values()
        self.assertEqual([], rnglst)
        indices = cpx.linear_constraints.add(names=["c0", "c1"])
        rnglst = cpx.linear_constraints.get_range_values()
        self.assertEqual([0.0, 0.0], rnglst)

    def testGetSetRangeValuesByIndex(self):
        with self._newCplex() as cpx:
            indices = cpx.linear_constraints.add(names=["c0", "c1"])
            self.getSetRangeValuesIndividual(cpx, indices)

    def testGetSetRangeValuesByName(self):
        varnames = ["c0", "c1"]
        with self._newCplex() as cpx:
            cpx.linear_constraints.add(names=varnames)
            self.getSetRangeValuesIndividual(cpx, varnames)

    def getSetRangeValuesIndividual(self, cpx, seq):
        for i, item in enumerate(seq):
            rng = cpx.linear_constraints.get_range_values(item)
            self.assertEqual(0.0, rng)
            newrng = i + 1.0
            cpx.linear_constraints.set_range_values(item, newrng)
            rng = cpx.linear_constraints.get_range_values(item)
            self.assertEqual(newrng, rng)

    def testGetSetRangeValuesByIndexList(self):
        with self._newCplex() as cpx:
            indices = cpx.linear_constraints.add(names=["c0", "c1"])
            indlst = list(indices)
            self.getSetRangeValuesSequence(cpx, indlst)

    def testGetSetRangeValuesByNameList(self):
        varnames = ["c0", "c1"]
        with self._newCplex() as cpx:
            cpx.linear_constraints.add(names=varnames)
            self.getSetRangeValuesSequence(cpx, varnames)

    def testGetSetRangeValuesByMixedList(self):
        varnames = ["c0", "c1"]
        with self._newCplex() as cpx:
            indlst = list(cpx.linear_constraints.add(names=varnames))
            self.getSetRangeValuesSequence(cpx, [indlst[0], varnames[-1]])

    def getSetRangeValuesSequence(self, cpx, seq):
        rnglst = cpx.linear_constraints.get_range_values(seq)
        self.assertEqual([0.0, 0.0], rnglst)
        cpx.linear_constraints.set_range_values(
            [(ind, i + 1.0) for i, ind in enumerate(seq)])
        expected = [1.0, 2.0]
        rnglst = cpx.linear_constraints.get_range_values(seq)
        self.assertEqual(expected, rnglst)
        # Now, try in reverse order
        expected.reverse()
        revseq = list(reversed(seq))
        rnglst = cpx.linear_constraints.get_range_values(revseq)
        self.assertEqual(expected, rnglst)

    def testGetRangeValuesBeginEndByIndex(self):
        with self._newCplex() as cpx:
            indlst = list(cpx.linear_constraints.add(
                names=["c0", "c1", "c2"]))
            self.assertEqual(3, len(indlst))
            self.getSetRangeValuesBeginEnd(cpx, indlst)

    def testGetRangeValuesBeginEndByName(self):
        varnames = ["c0", "c1", "c2"]
        with self._newCplex() as cpx:
            cpx.linear_constraints.add(names=varnames)
            self.getSetRangeValuesBeginEnd(cpx, varnames)

    def testGetRangeValuesBeginEndByMix(self):
        varnames = ["c0", "c1", "c2"]
        with self._newCplex() as cpx:
            indlst = list(cpx.linear_constraints.add(names=varnames))
            self.assertEqual(3, len(indlst))
            indlst[-1] = varnames[-1]
            self.getSetRangeValuesBeginEnd(cpx, indlst)

    def getSetRangeValuesBeginEnd(self, cpx, seq):
        begin = seq[0]
        end = seq[-1]
        rnglst = cpx.linear_constraints.get_range_values(begin, end)
        self.assertEqual([0.0, 0.0, 0.0], rnglst)
        cpx.linear_constraints.set_range_values(
            [(ind, i + 1.0) for i, ind in enumerate(seq)])
        rnglst = cpx.linear_constraints.get_range_values(begin, end)
        self.assertEqual([1.0, 2.0, 3.0], rnglst)

    def testSetSensesSingle(self):
        cpx = self._newCplex()
        cpx.variables.add(lb=[0.0]*len(sensetypes))
        cpx.linear_constraints.add(rhs=[0.0]*len(sensetypes))
        for idx, sense in enumerate(sensetypes):
            # By default, sense should be 'E'
            self.assertEqual(cpx.linear_constraints.get_senses(idx), 'E')
            cpx.linear_constraints.set_senses(idx, sense)
            self.assertEqual(cpx.linear_constraints.get_senses(idx), sense)

    def testSetSensesMultiple(self):
        cpx = self._newCplex()
        cpx.variables.add(lb=[0.0]*len(sensetypes))
        cpx.linear_constraints.add(rhs=[0.0]*len(sensetypes))
        cpx.linear_constraints.set_senses([(idx, sense) for idx, sense
                                           in enumerate(sensetypes)])
        for idx, sense in enumerate(sensetypes):
            self.assertEqual(cpx.linear_constraints.get_senses(idx), sense)

    def getZeroMatrix(self, num_var=3, num_con=3):
        """Build a coefficients matrix of zeros.

        num_var - the number of variables.
        num_con - the number of linear constraints.
        """
        cpx = self._newCplex()
        varnames = ["x{0}".format(i) for i in range(num_var)]
        varind = list(cpx.variables.add(names=varnames))
        connames = ["c{0}".format(i) for i in range(num_con)]
        conind = list(cpx.linear_constraints.add(names=connames))
        # Should be all zeros to begin with (i.e., empty lists).
        for r in cpx.linear_constraints.get_rows():
            ind, val = r.unpack()
            self.assertEqual(ind, [])
            self.assertEqual(val, [])
        return ZeroMatrix(cpx, varind, varnames, conind, connames)

    def testSetLinearComponentsByName(self):
        """Change one constraint at a time by name."""
        cpx, varind, varnames, conind, connames = self.getZeroMatrix()
        nvars = len(varind)
        for c in connames:
            cpx.linear_constraints.set_linear_components(
                c,
                SparsePair(ind=varnames, val=[1.0] * nvars)
            )
        for r in cpx.linear_constraints.get_rows():
            ind, val = r.unpack()
            self.assertEqual(ind, varind)
            self.assertEqual(val, [1.0] * nvars)

    def testSetLinearComponentsByIndex(self):
        """Change one constraint at a time by index."""
        cpx, varind, varnames, conind, connames = self.getZeroMatrix()
        nvars = len(varind)
        for c in conind:
            cpx.linear_constraints.set_linear_components(
                c,
                SparsePair(ind=varind, val=[2.0] * nvars)
            )
        for r in cpx.linear_constraints.get_rows():
            ind, val = r.unpack()
            self.assertEqual(ind, varind)
            self.assertEqual(val, [2.0] * nvars)

    def testSetLinearComponentsBatchByName(self):
        """Change all constraints at once by name."""
        cpx, varind, varnames, conind, connames = self.getZeroMatrix()
        nvars = len(varind)
        seq_of_pairs = []
        for c in connames:
            seq_of_pairs.append((c, SparsePair(ind=varnames,
                                               val=[1.0] * nvars)))
        cpx.linear_constraints.set_linear_components(seq_of_pairs)
        for r in cpx.linear_constraints.get_rows():
            ind, val = r.unpack()
            self.assertEqual(ind, varind)
            self.assertEqual(val, [1.0] * nvars)

    def testSetLinearComponentsBatchByIndex(self):
        """Change all constraints at once by index."""
        cpx, varind, varnames, conind, connames = self.getZeroMatrix()
        nvars = len(varind)
        seq_of_pairs = []
        for c in conind:
            seq_of_pairs.append((c, SparsePair(ind=varind,
                                               val=[2.0] * nvars)))
        cpx.linear_constraints.set_linear_components(seq_of_pairs)
        for r in cpx.linear_constraints.get_rows():
            ind, val = r.unpack()
            self.assertEqual(ind, varind)
            self.assertEqual(val, [2.0] * nvars)

    def testSetLinearComponentsNoArgs(self):
        """Test with no arguments."""
        cpx, varind, varnames, conind, connames = self.getZeroMatrix()
        nvars = len(varind)
        with self.assertRaises(WrongNumberOfArgumentsError):
            cpx.linear_constraints.set_linear_components()

    def testSetLinearComponentsEmptyList(self):
        """Test with empty list."""
        cpx, varind, varnames, conind, connames = self.getZeroMatrix()
        nvars = len(varind)
        # We expect that an error is not raised and that the return value
        # of set_linear_components() is None.
        self.assertIsNone(cpx.linear_constraints.set_linear_components([]))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
