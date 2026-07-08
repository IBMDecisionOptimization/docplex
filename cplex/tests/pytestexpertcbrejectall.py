#!/usr/bin/python
import sys
import unittest

from cplextestcase import CplexTestCase
import cplex
from cplex.callbacks import Context

RELDATADIR = '../../../examples/data'

class MyCallback():

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.calls = 0

    def invoke(self, context):
        self.calls += 1
        try:
            assert context.in_candidate(), "not in candidate"
            assert context.get_id() == Context.id.candidate, \
                "get_id() != candidate"
            assert not bool(context.get_int_info(context.info.feasible)), \
                "has feasible"
            assert context.is_candidate_point()
            assert not context.is_candidate_ray()
            if self.verbose:
                print("Thread ID: {0}".format(
                    context.get_int_info(context.info.thread_id)))
                print("Rejecting candidate objective: {0}".format(
                    context.get_candidate_objective()))
            # Reject all candidates.
            context.reject_candidate()
        except BaseException as err:
            print(err)
            raise


class ExpertCallbackRejectAllTests(CplexTestCase):
    def _test_model(self, modelfile):
        with cplex.Cplex(modelfile) as c:
            c = cplex.Cplex()
            c.read(modelfile)
            cb = MyCallback(verbose=True)
            contextmask = 0
            contextmask |= Context.id.candidate
            c.set_callback(cb, Context.id.candidate)
            c.parameters.dettimelimit.set(10000)
            c.parameters.mip.limits.nodes.set(1000)
            c.solve()
            print("Status:", c.solution.get_status_string())
            print("Callback invocations:", cb.calls)
            self.assertGreaterEqual(cb.calls, 100)

    def test_aflow30(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz')

if __name__ == '__main__':
    unittest.main()
