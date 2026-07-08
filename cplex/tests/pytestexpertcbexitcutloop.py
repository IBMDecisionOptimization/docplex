# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2019, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Test Cplex.callback.Context.exit_cut_loop

We test by solving twice: once without doing anything in the
callback and once by always exiting immediately. Then we compare
the number of active cuts at the end of the solve in either case
and require that in the second case fewer cuts were separated.
"""
from collections import namedtuple
import unittest
import random
import sys
import traceback

from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
import cplex

RELDATADIR = '../../../examples/data'
CutCounts = namedtuple("CutCounts", ["count", "sum"])


class ExpertCallback():
    def __init__(self, stop):
        self.stop = stop

    def invoke(self, context):
        if context.in_relaxation():
            if self.stop:
                context.exit_cut_loop()
        else:
            try:
                context.exit_cut_loop()
            except CplexSolverError as cse:
                assert cse.args[2] == error_codes.CPXERR_UNSUPPORTED_OPERATION

class ExpertCallbackExitCutLoop(CplexTestCase):
    def _solve_model(self, modelfile, stop, nodelim):
        with cplex.Cplex(modelfile) as cpx:
            cpx.parameters.mip.limits.nodes.set(nodelim)
            cb = ExpertCallback(stop)
            cpx.set_callback(cb, -1)
            cpx.solve()
            count = {c: cpx.solution.MIP.get_num_cuts(c)
                     for c in cpx.solution.MIP.cut_type}
            info = CutCounts(count=count, sum=sum(count.values()))
            return info
    def _test_model(self, modelfile, nodelim):
        ref = self._solve_model(modelfile, False, nodelim)
        assert ref.sum > 0

        stop = self._solve_model(modelfile, True, nodelim)
        assert ref.sum > stop.sum
        for k in sorted(ref.count):
            assert ref.count[k] >= stop.count[k]

    def test_aflow30(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz', 100)


if __name__ == '__main__':
    unittest.main()
