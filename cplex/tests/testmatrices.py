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
Tests linear_constraints and variables with unicode strings.

No command line arguments are required.
"""
import unittest
import sys
import cplex
from cplextestcase import CplexTestCase
from cplex.exceptions import CplexError
from cplex import SparsePair, SparseTriple
from cplex._internal._procedural import chbmatrix

UTF8 = 'utf-8'


class SparseTripleTests(CplexTestCase):

    def testDefaultArgs(self):
        """Test that default args are not mutable."""
        for i in range(3):
            st = SparseTriple()
            st.ind1.append(i)
            st.ind2.append(i)
            st.val.append(float(i))
            ind1, ind2, val = st.unpack()
            self.assertEqual([i], ind1)
            self.assertEqual([i], ind2)
            self.assertEqual([float(i)], val)

    def testEmpty(self):
        st = SparseTriple()
        self.assertTrue(st.isvalid())
        ind1, ind2, val = st.unpack()
        self.assertEqual([], ind1)
        self.assertEqual([], ind2)
        self.assertEqual([], val)
        self.assertEqual(
            "SparseTriple(ind1 = [], ind2 = [], val = [])",
            repr(st))

    def testNonEmpty(self):
        st = SparseTriple(ind1=[0, 1], ind2=[0, 1], val=[0, 0])
        self.assertTrue(st.isvalid())
        self.assertEqual(
            "SparseTriple(ind1 = [0, 1], ind2 = [0, 1], val = [0, 0])",
            repr(st))

    def testBadInput(self):
        try:
            st = SparseTriple(ind1=[0], ind2=[0, 1], val=[0, 1])
            self.fail()
        except CplexError as ce:
            self.assertIn("Inconsistent input data to SparseTriple",
                          str(ce))

    def testUnpack(self):
        st = SparseTriple(ind1=[0, 1], ind2=[1, 2], val=[0, 0])
        ind1, ind2, val = st.unpack()
        self.assertEqual([0, 1], ind1)
        self.assertEqual([1, 2], ind2)
        self.assertEqual([0, 0], val)

    def testIsValid(self):
        st = SparseTriple(ind1=[0], ind2=[0], val=[0])
        self.assertTrue(st.isvalid())
        st.val.append(1)
        self.assertFalse(st.isvalid())


class SparsePairTests(CplexTestCase):

    def testDefaultArgs(self):
        """Test that default args are not mutable."""
        for i in range(3):
            sp = SparsePair()
            sp.ind.append(i)
            sp.val.append(float(i))
            ind, val = sp.unpack()
            self.assertEqual([i], ind)
            self.assertEqual([float(i)], val)

    def testEmpty(self):
        sp = SparsePair()
        self.assertTrue(sp.isvalid())
        ind, val = sp.unpack()
        self.assertEqual([], ind)
        self.assertEqual([], val)
        self.assertEqual("SparsePair(ind = [], val = [])",
                         repr(sp))

    def testNonEmpty(self):
        sp = SparsePair(ind=[0, 1], val=[0, 0])
        self.assertTrue(sp.isvalid())
        self.assertEqual("SparsePair(ind = [0, 1], val = [0, 0])",
                         repr(sp))

    def testBadInput(self):
        try:
            sp = SparsePair(ind=[0], val=[0, 1])
            self.fail()
        except CplexError as ce:
            self.assertTrue(
                "Inconsistent input data to SparsePair" in str(ce))

    def testUnpack(self):
        sp = SparsePair(ind=[0, 1], val=[0, 0])
        ind, val = sp.unpack()
        self.assertEqual([0, 1], ind)
        self.assertEqual([0, 0], val)

    def testIsValid(self):
        sp = SparsePair(ind=[0], val=[0])
        self.assertTrue(sp.isvalid())
        sp.val.append(1)
        self.assertFalse(sp.isvalid())

class Foo():
    pass

class CHBMatrixTests(CplexTestCase):

    def testBadCoefMatrixSeq(self):
        cpx = self._newCplex()
        try:
            with chbmatrix(Foo(), cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except TypeError as te:
            self.assertTrue(
                "coefficient matrix must be a sequence" in str(te))

    def testBadCoefMatrixLen(self):
        cpx = self._newCplex()
        try:
            with chbmatrix([[[0,0]]], cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except IndexError as ie:
            self.assertTrue(
                "sequence elements of coefficient matrix "
                "must have length 2" in str(ie))

    def testBadRowItem(self):
        cpx = self._newCplex()
        try:
            with chbmatrix([[Foo()]], cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except IndexError as ie:
            self.assertTrue(
                "sequence elements of coefficient matrix "
                "must have length 2" in str(ie))

    def testBadSparsePairMember(self):
        cpx = self._newCplex()
        try:
            with chbmatrix([[Foo(), Foo()]], cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except TypeError as exc:
            self.assertTrue(
                "object of type 'Foo' has no len()" in str(exc))

    def testBadCoefMatrixElem(self):
        cpx = self._newCplex()
        try:
            with chbmatrix([Foo()], cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except TypeError as exc:
            self.assertTrue(
                "elements of coefficient matrix must be sequences "
                "or instances of cplex.SparsePair" in str(exc))

    def testBadNameInSequenceInd(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['a', 'b'])
        try:
            with chbmatrix([[['a', 'c'], [0, 0]]],
                           cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except ValueError as exc:
            self.assertTrue(
                " 1210: Invalid name -- 'c'" in str(exc))

    def testBadNameInSparsePairInd(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['a', 'b'])
        try:
            with chbmatrix([SparsePair(ind=['a', 'c'], val=[0, 0])],
                           cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except ValueError as exc:
            self.assertTrue(
                " 1210: Invalid name -- 'c'" in str(exc))

    def testBadTypeInSequenceInd(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['a', 'b'])
        try:
            with chbmatrix([[['a', Foo()], [0, 0]]],
                           cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except ValueError as exc:
            self.assertTrue(
                " invalid matrix input type --" in str(exc))
            self.assertEqual(2, len(exc.args))
            self.assertTrue(isinstance(exc.args[1], Foo))

    def testBadTypeInSparsePairInd(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['a', 'b'])
        try:
            with chbmatrix([SparsePair(ind=['a', Foo()], val=[0, 0])],
                           cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except ValueError as exc:
            self.assertTrue(
                " invalid matrix input type --" in str(exc))
            self.assertEqual(2, len(exc.args))
            self.assertTrue(isinstance(exc.args[1], Foo))

    def testBadPairInSequence(self):
        cpx = self._newCplex()
        try:
            with chbmatrix([[[0, 1], [0]]], cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except IndexError as exc:
            self.assertTrue(
                "sequence elements of coefficient matrix must "
                "contain sequences of equal length" in str(exc))

    def testBadPairInSparsePair(self):
        cpx = self._newCplex()
        try:
            sp = SparsePair(ind=[0], val=[0])
            sp.ind.append(1)  # Purposely corrupt it
            with chbmatrix([sp], cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except IndexError as exc:
            self.assertTrue(
                "sequence elements of coefficient matrix must "
                "contain sequences of equal length" in str(exc))

    def testBadTypeInSequenceVal(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['a', 'b'])
        try:
            with chbmatrix([[['a', 1], [0, Foo()]]],
                           cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except ValueError as exc:
            self.assertTrue(
                " invalid matrix input type --" in str(exc))
            self.assertEqual(2, len(exc.args))
            self.assertTrue(isinstance(exc.args[1], Foo))

    def testBadTypeInSparsePairVal(self):
        cpx = self._newCplex()
        cpx.variables.add(names=['a', 'b'])
        try:
            with chbmatrix([[['a', 1], [0, Foo()]]],
                           cpx._env_lp_ptr, 0):
                pass
            self.fail()
        except ValueError as exc:
            self.assertTrue(
                " invalid matrix input type --" in str(exc))
            self.assertEqual(2, len(exc.args))
            self.assertTrue(isinstance(exc.args[1], Foo))

    def testInWithStmt(self):
        with self._newCplex() as cpx:
            with chbmatrix([[[0, 1], [1., 1.]]],
                           cpx._env_lp_ptr, 0) as (mat, nnz):
                matbeg, matind, matval = mat
                self.assertTrue(isinstance(matbeg, int))
                self.assertTrue(isinstance(matind, int))
                self.assertTrue(isinstance(matval, int))
                self.assertEqual(2, nnz)

    def testWithTuples(self):
        with self._newCplex() as cpx:
            with chbmatrix((((0, 1), (1., 1.)),),
                           cpx._env_lp_ptr, 0) as (mat, nnz):
                matbeg, matind, matval = mat
                self.assertEqual(nnz, 2)

    def testWithNamesAndIndices(self):
        varnames = ['a', 'b']
        with self._newCplex() as cpx:
            indlst = list(cpx.variables.add(names=varnames))
            with chbmatrix([[[indlst[0], varnames[1]], [1., 2.]]],
                           cpx._env_lp_ptr, 0) as (mat, nnz):
                matbeg, matind, matval = mat
                self.assertEqual(nnz, 2)

    def testWithSparsePairNamesAndIndices(self):
        varnames = ['a', 'b']
        with self._newCplex() as cpx:
            indlst = list(cpx.variables.add(names=varnames))
            with chbmatrix([SparsePair(ind=[indlst[0], varnames[1]],
                                       val=[1., 2.])],
                           cpx._env_lp_ptr, 0) as (mat, nnz):
                matbeg, matind, matval = mat
                self.assertEqual(nnz, 2)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
