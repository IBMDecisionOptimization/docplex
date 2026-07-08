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
Tests the Cplex.start interface.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase

OBJECTIVE_EPSILON = 1e-7


class StartTests(CplexTestCase):

    # This test results in some strange behavior:
    ## In the log for cpx3, first we see that no start values are
    ## read:
    # ...
    # Warning, line 31: 'x1' is not a variable.
    # Warning, line 32: 'x2' is not a variable.
    # Warning, line 33: 'x3' is not a variable.
    # Warning, line 34: 'x4' is not a variable.
    # Warning: no MIP start values read, no MIP start loaded.
    # ...
    ## Then we see that an initial solution was defined:
    # MIP start 'incumbent' defined solution with objective -205.4697.
    # 1 of 1 MIP starts provided solutions.
    # MIP start 'incumbent' defined initial solution with objective -205.4697.
    # ...
    ## Finally, we fail our equality check at the end.
    @unittest.skip("FIXME")
    def testBogusStart(self):
        with self._newCplex() as cpx1, \
                 self._newCplex() as cpx2, \
                 self._newCplex() as cpx3, \
                 self._getTempFileName(ext=".sol") as tmp:
            # First, solve problem A.
            cpx1.read(self._getResource("examples/data/mexample.mps"))
            cpx1.solve()
            cpx1.solution.write(tmp)
            # Second, solve problem B.
            cpx2.read("../../data/caso8.mps")
            cpx2.solve()
            # Now apply unrelated solution A to problem B.
            cpx3.read("../../data/caso8.mps")
            cpx3.start.read_start(tmp)
            cpx3.solve()
            # Expecting unrelated start has no effect on final solution.
            self.assertAlmostEqual(cpx2.solution.get_objective_value(),
                                   cpx3.solution.get_objective_value(),
                                   delta=1e-4)

    def testAfiroBarrier(self):
        with self._newCplex() as dummy:
            self._testReadStart(
                "../../data/afiro.mps",
                lpmethod=dummy.parameters.lpmethod.values.barrier
            )

    def testAfiroDual(self):
        with self._newCplex() as dummy:
            self._testReadStart(
                "../../data/afiro.mps",
                lpmethod=dummy.parameters.lpmethod.values.dual
            )

    def testAfiroPrimal(self):
        with self._newCplex() as dummy:
            self._testReadStart(
                "../../data/afiro.mps",
                lpmethod=dummy.parameters.lpmethod.values.primal
            )

    def testCaso8(self):
        self._testReadStart(self._getResource("examples/data/caso8.mps"))

    def testAflow30a(self):
        self._testReadStart(self._getResource("examples/data/aflow30a.mps.gz"))

    def testMExample(self):
        self._testReadStart(self._getResource("examples/data/mexample.mps"))

    def _testReadStart(self, modelfile, lpmethod=None,
                       epsilon=OBJECTIVE_EPSILON):
        with self._newCplex() as cpx1, \
                 self._newCplex() as cpx2, \
                 self._getTempFileName(ext=".sol") as tmp:
            # If recording, then skip this test because we use temp
            # files.
            self.skipIfParamTesting(cpx1)

            if lpmethod:
                cpx1.parameters.lpmethod.set(lpmethod)
                cpx2.parameters.lpmethod.set(lpmethod)
            cpx1.parameters.mip.tolerances.mipgap.set(0.0)
            cpx2.parameters.mip.tolerances.mipgap.set(0.0)
            cpx1.read(modelfile)

            if cpx1.get_problem_type() == cpx1.problem_type.MILP:
                self.assertEqual(0, cpx1.MIP_starts.get_num())

            cpx1.solve()
            cpx1.solution.write(tmp)
            cpx2.read(modelfile)
            cpx2.start.read_start(tmp)

            if cpx2.get_problem_type() == cpx2.problem_type.MILP:
                self.assertGreater(cpx2.MIP_starts.get_num(), 0)

            cpx2.solve()
            self.assertAlmostEqual(cpx1.solution.get_objective_value(),
                                   cpx2.solution.get_objective_value(),
                                   delta=epsilon)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
