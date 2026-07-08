# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2018, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Simple tests for querying Benders cuts from info callback
"""
import unittest
from cplex.callbacks import MIPInfoCallback
from cplextestcase import CplexTestCase

RELDATADIR = '../../../examples/data'

class InfoCallback(MIPInfoCallback):
    def __call__(self):
        self.called = True
        count = self.get_num_cuts(MIPInfoCallback.cut_type.benders)
        if count > 0:
            self.count = True

class TestQueryBendersCuts(CplexTestCase):
    def setUp(self):
        """Runs before every test."""
        self.cpx = self._newCplex()
        self.cpx.read(RELDATADIR + "/UFL_25_35_1.mps")
        self.cpx.read_annotations(RELDATADIR + "/UFL_25_35_1.ann")
    def tearDown(self):
        self.cpx.end()

    def testQuery(self):
        cpx = self.cpx
        cb = cpx.register_callback(InfoCallback)
        cb.called = False
        cb.count = False
        cpx.parameters.mip.limits.nodes.set(1000)
        cpx.solve()
        self.assertTrue(cb.called)
        self.assertTrue(cb.count);


if __name__ == '__main__':
    unittest.main()
