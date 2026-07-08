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
Tests callback setup.

No command line arguments are required.
"""
import unittest
import traceback
from cplextestcase import CplexTestCase
from cplex.callbacks import IncumbentCallback


class SimpleCallback(IncumbentCallback):

    def __init__(self, env):
        super().__init__(env)
        self.was_called = False

    def __call__(self):
        self.was_called = True
        try:
            self.get_num_cols()
        except Exception:
            traceback.print_exc()
            raise
        self.abort()


class CallbackSetupTests(CplexTestCase):

    def setUp(self):
        self.cpx = self._newCplex()
        self.cpx.read("../../data/caso8.mps")
        self.cb = self.cpx.register_callback(SimpleCallback)

    def tearDown(self):
        self.cpx.end()

    def testSolve(self):
        self.cpx.solve()
        self.assertTrue(self.cb.was_called)

    def testRunseeds(self):
        self.cpx.runseeds(cnt=1)
        self.assertTrue(self.cb.was_called)

    def testPopulate(self):
        self.cpx.populate_solution_pool()
        self.assertTrue(self.cb.was_called)

    def testFeasopt(self):
        self.cpx.feasopt(self.cpx.feasopt.all_constraints())
        self.assertTrue(self.cb.was_called)

    def testConflictRefine(self):
        with self._newCplex() as cpx:
            cpx.parameters.dettimelimit.set(100.0)
            cpx.read("../../data/infmip.lp")
            cb = cpx.register_callback(SimpleCallback)
            cpx.conflict.refine(cpx.conflict.all_constraints())
            self.assertTrue(cb.was_called)

    def testConflictRefineMIPStart(self):
        self.cpx.MIP_starts.add([[0], [5000.0]],
                                self.cpx.MIP_starts.effort_level.auto)
        self.cpx.conflict.refine_MIP_start(
            0, self.cpx.conflict.all_constraints())
        self.assertTrue(self.cb.was_called)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
