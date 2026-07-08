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
Tests the Cplex.runseeds method.

No command line arguments are required.
"""
import unittest
import os
import re

from cplextestcase import CplexTestCase
from testcplex import KeyboardInterruptCallback
from cplex.exceptions import (CplexError, CplexSolverError, error_codes)
from cplex._internal import ProblemType
from cplex import Aborter
from testutil import OutputProcessor


class RunseedsTests(CplexTestCase):

    def testKeyboardInterrupt(self):
        """Simulate a Ctrl-C event during runseeds."""
        # Signal handling does not work on Windows when run via cygwin,
        # so we skip this test there.
        if self.iswindows():
            return

        with self._newCplex() as cpx:
            # Prevent long running process when testing the recorder.
            # Callbacks are skipped.
            self.skipIfParamTesting(cpx)
            cpx.read(self._getResource("examples/data/noswot.mps"))
            cb = cpx.register_callback(KeyboardInterruptCallback)
            cb.pid = os.getpid()
            cpx.runseeds(cnt=30)
            # In opportunistic mode and on AIX, we consistently get
            # cb.numkilled == 2 or 3, so we relax the assertions below
            # (see RTC-37310). The point of this test is to show that we
            # can kill runseeds and that we do not have to kill every
            # solve (i.e., cnt=30 above).
            #self.assertEqual(cb.numkilled, 1)
            self.assertGreaterEqual(cb.numkilled, 1)
            self.assertGreaterEqual(3, cb.numkilled)

    def testWithAborter(self):
        # Prevent long running process when testing the recorder.
        # Termination via an aborter is skipped.
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            cpx.read(self._getResource("examples/data/noswot.mps"))
            aborter = Aborter()
            cpx.use_aborter(aborter)
            aborter.abort()
            cpx.runseeds()

    def testWithZeroCount(self):
        with self._newCplex() as cpx:
            cpx.read("../../data/caso8.mps")
            try:
                cpx.runseeds(cnt=0)
                self.fail()
            except CplexSolverError as err:
                self.assertEqual(err.args[2], error_codes.CPXERR_BAD_ARGUMENT)

    def testWithNegativeCount(self):
        with self._newCplex() as cpx:
            cpx.read("../../data/caso8.mps")
            try:
                cpx.runseeds(cnt=-1)
                self.fail()
            except CplexSolverError as err:
                self.assertEqual(err.args[2], error_codes.CPXERR_BAD_ARGUMENT)

    def testWithInfeasible(self):
        with self._newCplex() as cpx:
            cpx.read(self._getResource("examples/data/infeasible.lp"))
            cpx.runseeds(cnt=1)
            # There's nothing to check here ... it would be nice if we
            # could query the statistics....

    def testLP(self):
        self.checkUnsupportedProblemType(
            ProblemType.LP,
            error_codes.CPXERR_UNSUPPORTED_OPERATION,
            "../../data/afiro.mps")

    def testQCP(self):
        self.checkUnsupportedProblemType(
            ProblemType.QCP,
            error_codes.CPXERR_UNSUPPORTED_OPERATION,
            "../../data/qcp.lp")

    def testFixedMILP(self):
        with self._newCplex() as cpx:
            cpx.read("../../data/caso8.mps")
            cpx.solve()
            cpx.set_problem_type(ProblemType.fixed_MILP)
            self.checkUnsupportedProblemType(
                ProblemType.LP,
                error_codes.CPXERR_UNSUPPORTED_OPERATION)

    def testFixedMILP2(self):
        with self._newCplex() as cpx:
            cpx.read("../../data/caso8.mps")
            cpx.solve()
            cpx.set_problem_type(ProblemType.fixed_MILP)
            try:
                cpx.runseeds(cnt=1)
                self.fail()
            except CplexSolverError as err:
                self.assertEqual(err.args[2],
                                 error_codes.CPXERR_UNSUPPORTED_OPERATION)

    def testNodeLP(self):
        pass  # FIXME: Not implemented

    def testQP(self):
        pass  # FIXME: Not implemented

    def testFixedMIQP(self):
        pass  # FIXME: Not implemented

    def testNodeQP(self):
        pass  # FIXME: Not implemented

    def testNodeQCP(self):
        pass  # FIXME: Not implemented

    def testMILP(self):
        self.checkSupportedProblemType(
            "../../data/caso8.mps",
            ProblemType.MILP)

    def testMIQP(self):
        self.checkSupportedProblemType(
            "../../data/miqp0033.mps",
            ProblemType.MIQP)

    def testMIQCP(self):
        pass  # FIXME: Not implemented

    def checkSupportedProblemType(self, model_path, problem_type):
        with self._newCplex() as cpx:
            cpx.read(model_path)
            self.assertEqual(problem_type, cpx.get_problem_type())
            cpx.runseeds(cnt=2)

    def checkUnsupportedProblemType(self, problem_type, status,
                                    model_path=None, cpx=None):
        if cpx is None:
            cpx = self._newCplex()
        if model_path is not None:
            cpx.read(model_path)
        self.assertEqual(problem_type, cpx.get_problem_type())
        try:
            cpx.runseeds(cnt=1)
            self.fail()
        except CplexSolverError as err:
            self.assertEqual(err.args[2], status)

    def testOutput(self):
        with self._newCplex() as cpx:
            rsproc = OutputProcessor(
                ["Starting variability optimization #1",
                 "Starting variability optimization #2",
                 "runseeds statistics of 2 runs"])
            cpx.set_results_stream(rsproc)
            cpx.read("../../data/caso8.mps")
            cpx.runseeds(cnt=2)
            for item in rsproc.regex_list:
                self.assertEqual(item.num_matches, 1)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
