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
Tests Aborter functionality.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase
from cplex import Aborter


class AborterTests(CplexTestCase):

    def setUp(self):
        self.cpx = self._newCplex()

    def tearDown(self):
        self.cpx.end()

    def checkSolve(self, cpx, expected_status):
        if cpx is None:
            cpx = self.cpx
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(), expected_status)

    def checkSolveOptimal(self, cpx=None):
        self.checkSolve(cpx, self.cpx.solution.status.optimal)

    def checkSolveAborted(self, cpx=None):
        self.checkSolve(cpx, self.cpx.solution.status.abort_user)

    def testEmptyAborted(self):
        with Aborter() as aborter:
            aborter.abort()
            self.cpx.use_aborter(aborter)
            self.checkSolveAborted()

    def testEmptyNotAborted(self):
        with Aborter() as aborter:
            self.cpx.use_aborter(aborter)
            self.checkSolveOptimal()

    def testEmptyAfterClear(self):
        with Aborter() as aborter:
            aborter.abort()
            self.cpx.use_aborter(aborter)
            aborter.clear()
            self.checkSolveOptimal()

    def testEmptyAfterEnd(self):
        with Aborter() as aborter:
            aborter.abort()
            self.cpx.use_aborter(aborter)
            aborter.end()
            self.checkSolveOptimal()

    def testEmptyAfterUse(self):
        with Aborter() as aborter1:
            aborter1.abort()
            self.cpx.use_aborter(aborter1)
            with Aborter() as aborter2:
                self.cpx.use_aborter(aborter2)
                self.checkSolveOptimal()

    def testRemoveNone(self):
        self.assertIsNone(self.cpx.remove_aborter())

    def testMultiple(self):
        with self._newCplex() as cpx1:
            with self._newCplex() as cpx2:
                with Aborter() as aborter:
                    cpx1.use_aborter(aborter)
                    cpx2.use_aborter(aborter)
                    aborter.abort()
                    self.checkSolveAborted(cpx1)
                    self.checkSolveAborted(cpx2)

    def testGetAborterOnEmpty(self):
        self.assertIsNone(self.cpx.get_aborter())

    def testGetAborterAfterRemove(self):
        with Aborter() as aborter:
            self.cpx.use_aborter(aborter)
            self.assertEqual(self.cpx.get_aborter(), aborter)
            self.cpx.remove_aborter()
            self.assertIsNone(self.cpx.get_aborter())

    def testIsAborted(self):
        with Aborter() as aborter:
            self.assertFalse(aborter.is_aborted())
            aborter.abort()
            self.assertTrue(aborter.is_aborted())
            aborter.clear()
            self.assertFalse(aborter.is_aborted())

    def testEnd(self):
        with Aborter() as aborter:
            # Make sure nothing bad happens if called again.
            aborter.end()

    def testUseReturnValue(self):
        aborter = self.cpx.use_aborter(Aborter())
        self.assertEqual(self.cpx.get_aborter(), aborter)
        self.assertEqual(self.cpx.remove_aborter(), aborter)

    def testDel(self):
        aborter = Aborter()
        del aborter

    def testAfterCplexEnd(self):
        with Aborter() as aborter:
            with self._newCplex() as cpx:
                cpx.use_aborter(aborter)

    def checkValueError(self, func):
        try:
            func()
            self.fail()
        except ValueError:
            pass

    def testAfterEnd(self):
        aborter = Aborter()
        aborter.end()
        self.checkValueError(aborter.abort)
        self.checkValueError(aborter.clear)
        self.checkValueError(aborter.is_aborted)

    def testUseTwice(self):
        a1 = Aborter()
        a2 = Aborter()
        self.cpx.use_aborter(a1)
        self.cpx.use_aborter(a2)
        a1.end()
        a2.abort()
        self.checkSolveAborted()

    def testUseNone(self):
        aborter = self.cpx.use_aborter(None)
        self.assertIsNone(aborter)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
