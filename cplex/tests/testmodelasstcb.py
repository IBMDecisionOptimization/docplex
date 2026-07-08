# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Simple tests for the modeling assistance callback.
"""
import unittest

from cplextestcase import CplexTestCase
from cplex.exceptions import CplexSolverError, error_codes
from cplex import Aborter, model_info

RELDATADIR = '../../../examples/data'
EXPECTED_WARNING = ("CPLEX Warning  1048: Detected constraint with wide"
                    " range of coefficients. In constraint 'REST0233'"
                    " the ratio of largest and smallest (in absolute"
                    " value) coefficients is 1.23961e+09.")


class Callback():

    def __init__(self, expectedid, expectedmsg):
        self.calls = 0
        self.expectedid = expectedid
        self.expectedmsg = expectedmsg
        self.matchedid = False
        self.matchedmsg = False

    def invoke(self, issueid, message):
        self.calls += 1
        if self.expectedid == issueid:
            self.matchedid = True
        if message.find(self.expectedmsg) >= 0:
            self.matchedmsg = True


class AbortCallback():

    def __init__(self, aborter):
        self.aborter = aborter

    def invoke(self, issueid, message):
        self.aborter.abort()


class ModelAsstCBTests(CplexTestCase):

    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        self.cpx.read(RELDATADIR + "/caso8.mps")
        self.cpx.parameters.read.datacheck.set(
            self.cpx.parameters.read.datacheck.values.assist)

    def testSimple(self):
        cb = Callback(model_info.CPXMI_WIDE_COEFF_RANGE,
                      EXPECTED_WARNING)
        self.cpx.set_modeling_assistance_callback(cb)
        self.cpx.solve()
        self.assertGreater(cb.calls, 0)
        self.assertTrue(cb.matchedid)
        self.assertTrue(cb.matchedmsg)

    def testClear(self):
        cb = Callback(model_info.CPXMI_WIDE_COEFF_RANGE,
                      EXPECTED_WARNING)
        self.cpx.set_modeling_assistance_callback(cb)
        self.cpx.set_modeling_assistance_callback(None)
        self.cpx.solve()
        self.assertEqual(cb.calls, 0)
        self.assertFalse(cb.matchedid)
        self.assertFalse(cb.matchedmsg)

    def testAborted(self):
        aborter = self.cpx.use_aborter(Aborter())
        self.cpx.set_modeling_assistance_callback(AbortCallback(aborter))
        self.cpx.solve()
        self.assertEqual(self.cpx.solution.get_status(),
                         self.cpx.solution.status.MIP_abort_infeasible)


if __name__ == '__main__':
    unittest.main()
