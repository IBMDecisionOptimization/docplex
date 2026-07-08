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
Tests the ObjectiveInterface.

No command line arguments are required.
"""
import unittest
from cplex import SparsePair
from cplex.exceptions import (CplexSolverError,
                              WrongNumberOfArgumentsError,
                              error_codes)
from cplextestcase import CplexTestCase

SCORPION_PRESOLVE_SHIFT = 182.564202

class ObjectiveTests(CplexTestCase):

    def testGetOffsetEmpty(self):
        cpx = self._newCplex()
        self.assertEqual(0.0, cpx.objective.get_offset())

    def testSetOffset(self):
        cpx = self._newCplex()
        cpx.objective.set_offset(3.0)
        self.assertEqual(3.0, cpx.objective.get_offset())

    def testGetOffsetScorpion(self):
        cpx = self._newCplex()
        cpx.read(self._getResource("tests/data/scorpion.mps.gz"))
        with self._getTempFileName('.pre') as tmp:
            cpx.presolve.write(tmp)
            pre = self._newCplex()
            pre.read(tmp)
            self.assertEqual(SCORPION_PRESOLVE_SHIFT,
                             pre.objective.get_offset())

    def testGetOffsetScorpionWithConstant(self):
        cpx = self._newCplex()
        cpx.read(self._getResource("tests/data/scorpion.mps.gz"))
        cpx.objective.set_offset(3.0)
        cpx.presolve.presolve(cpx.presolve.method.dual)
        self.assertEqual(3.0, cpx.objective.get_offset())

    def testGetQuadraticEmpty(self):
        with self._newCplex() as cpx:
            self.assertEqual([], cpx.objective.get_quadratic())

    # Should passing in an empty list be a no-op?  This is sort of ugly
    # because set_quadratic has this code that either calls CPXXcopyqpsep
    # or CPXXcopyquad; it checks for a list of float's to determine which
    # on to call, but doesn't verify that the list is empty or not
    # beforehand.  We end up getting an index error or something like
    # that.  It seems like it would be better if we had two functions
    # instead of one (e.g., set_separable_quadratic and set_quadratic).
    @unittest.skip("FIXME?")
    def testSetQuadraticEmptyList(self):
        with self._newCplex() as cpx:
            cpx.objective.set_quadratic([])

    # Here, we end up converting empty lists to NULL's and pass those
    # into CPXXcopyquad.  This is related to testSetQuadraticEmptyList
    # in that it seems like it would be reasonble to interpret an empty
    # list as a no-op.  I don't like the idea of having special code in
    # Python layer to do that rather than handling it in the Callable
    # Library, but on the other hand, this is more of a Python style
    # issue, so maybe that would be acceptable.
    def testSetQuadraticEmptySparsePair(self):
        try:
            with self._newCplex() as cpx:
                cpx.objective.set_quadratic([SparsePair(ind=[], val=[])])
                self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NULL_POINTER)

    # We end up getting into the _HBMatrix constructor and an IndexError
    # is raised.  Seems like we should have some better error checking
    # there.
    @unittest.skip("FIXME?")
    def testSetQuadraticBadArg(self):
        with self._newCplex() as cpx:
            cpx.objective.set_quadratic(['a', 'b', 'c'])

    def testSetQuadraticWrongNumberOfArgs(self):
        try:
            with self._newCplex() as cpx:
                cpx.objective.set_quadratic()
                self.fail()
        except WrongNumberOfArgumentsError:
            pass

    def testGetQuadraticSeparable(self):
        numvars = 3
        with self._newCplex() as cpx:
            cpx.variables.add(lb=[0.0]*numvars)
            cpx.objective.set_quadratic([1.0]*numvars)
            actual = cpx.objective.get_quadratic()
            for i, sp in enumerate(actual):
                ind, val = sp.unpack()
                self.assertEqual([i], ind)
                self.assertEqual([1.0], val)

    def testGetQuadraticNotSeparable(self):
        numvars = 3
        indlst = [[0, 1, 2], [0, 1], [0, 2]]
        vallst = [[1.0, -2.0, 0.5], [-2.0, -1.0], [0.5, -3.0]]
        with self._newCplex() as cpx:
            cpx.variables.add(lb=[0.0]*numvars)
            cpx.objective.set_quadratic([SparsePair(ind=ind, val=val)
                                         for ind, val
                                         in zip(indlst, vallst)])
            actual = cpx.objective.get_quadratic()
            for i, sp in enumerate(actual):
                ind, val = sp.unpack()
                self.assertEqual(indlst[i], ind)
                self.assertEqual(vallst[i], val)

    def testGetQuadraticNotSeparableOneVar(self):
        with self._newCplex() as cpx:
            cpx.variables.add(lb=[0.0])
            cpx.objective.set_quadratic([SparsePair(ind=[0], val=[1.0])])
            [actual] = cpx.objective.get_quadratic()
            ind, val = actual.unpack()
            self.assertEqual([0], ind)
            self.assertEqual([1.0], val)

    def checkSetQuadraticSeparable(self, values):
        numvars = len(values)
        self.assertGreater(numvars, 1)
        with self._newCplex() as cpx:
            cpx.variables.add(lb=[0.0] * numvars)
            cpx.objective.set_quadratic(values)
            actual = cpx.objective.get_quadratic()
            for i, sp in enumerate(actual):
                ind, val = sp.unpack()
                self.assertEqual([i], ind)
                self.assertEqual([values[i]], val)

    def testSetQuadraticWithInts(self):
        self.checkSetQuadraticSeparable([1] * 3)

    def testSetQuadraticWithFloats(self):
        self.checkSetQuadraticSeparable([1.0] * 3)

    def testSetQuadraticWithMix(self):
        values = [1, 2.0, 3]
        self.checkSetQuadraticSeparable(values)

    # TODO: Implement tests for the remaining methods in ObjectiveInterface


def main():
    unittest.main()

if __name__ == '__main__':
    main()
