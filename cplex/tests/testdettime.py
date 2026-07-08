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
Tests deterministic time limits.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase

MIP_FILE = '../../data/noswot.mps'
LP_FILE = '../../data/afiro.mps'
INFEAS_FILE = '../../data/infeasible.lp'


class DetTimeTests(CplexTestCase):

    def testMipInfeasible(self):
        """Deterministic time limits with infeasible mipopt."""
        cpx = self._setUpCplex(MIP_FILE)
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.MIP_dettime_limit_infeasible)

    def testMipFeasible(self):
        """Deterministic time limites with feasible mipopt."""
        cpx = self._setUpCplex(MIP_FILE, dettimelimit=100)
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.MIP_dettime_limit_feasible)

    def testLP(self):
        """Deterministic time limits with lpopt."""
        cpx = self._setUpCplex(LP_FILE)
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.abort_dettime_limit)

    def testConflictAnalysis(self):
        """Deterministic time limits with conflict analysis."""
        cpx = self._setUpCplex(INFEAS_FILE, 0.001)
        cpx.conflict.refine(cpx.conflict.all_constraints())
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.conflict_abort_dettime_limit)

    def testTuning(self):
        """Deterministic time limits with tuning."""
        cpx = self._setUpCplex(INFEAS_FILE, 0.001)
        self.assertEqual(cpx.parameters.tune_problem(),
                         cpx.parameters.tuning_status.dettime_limit)

    def _setUpCplex(self, model_file, dettimelimit=0.01):
        cpx = self._newCplex()
        cpx.read(model_file)
        cpx.parameters.randomseed.set(0)
        cpx.parameters.dettimelimit.set(dettimelimit)
        return cpx


def main():
    unittest.main()

if __name__ == '__main__':
    main()
