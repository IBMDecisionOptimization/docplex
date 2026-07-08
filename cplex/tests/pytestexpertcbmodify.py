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
Simple test to check that a callback remains after loading a new model.
"""
import unittest
from cplextestcase import CplexTestCase
import cplex

RELDATADIR = '../../../examples/data'


class ExpertCallback():
    def __init__(self):
        self.calls = 0

    def invoke(self, context):
        self.calls += 1
        context.abort()


class ExpertCallbackModifyTests(CplexTestCase):
    def _test_model(self, modelfile):
        with cplex.Cplex(modelfile) as cpx:
            cb = ExpertCallback()
            cpx.set_callback(cb, -1)

            # Now read the model again, solve it and make sure the callback
            # was invoked (i.e., that loading a new model does not clear the
            # callback settings)
            cpx.read(modelfile)
            cpx.solve()
            print(cb.calls, 'callback invocations')
            self.assertGreaterEqual(cb.calls, 1)

    def test_aflow30a(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz')


if __name__ == '__main__':
    unittest.main()
