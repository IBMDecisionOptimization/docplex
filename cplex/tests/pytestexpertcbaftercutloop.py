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
Simple tests with an empty expert callback:
invoke all query functions in all contexts
"""
import unittest
import random
import sys
import traceback
from threading import Lock

from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
import cplex

RELDATADIR = '../../../examples/data'


class ExpertCallback():
    def __init__(self):
        self.calls = 0
        self.lastRestart = -1
        self.lck = Lock()
        self.nodes = dict()

    def invoke(self, context):
        try:
            restarts = context.get_int_info(
                cplex.callbacks.Context.info.restarts)
            with self.lck:
                self.calls += 1
                if restarts != self.lastRestart:
                    self.lastRestart = restarts;
                    self.nodes.clear()

            if context.in_relaxation():
                uid = context.get_long_info(cplex.callbacks.Context.info.node_uid)
                after = context.get_int_info(cplex.callbacks.Context.info.after_cut_loop)
                with self.lck:
                    # If this is an unknown node then insert it into the map
                    # with a false value
                    if uid not in self.nodes:
                        self.nodes[uid] = False

                    # If we saw "after cut loop" for this node before then we
                    # cannot get here
                    assert not self.nodes[uid]

                    # If cut loop is complete then mark the node
                    if after != 0:
                        self.nodes[uid] = True
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


class ExpertCallbackModifyTests(CplexTestCase):
    def _test_model(self, modelfile):
        with cplex.Cplex(modelfile) as cpx:
            cb = ExpertCallback()
            cpx.set_callback(cb, cplex.callbacks.Context.id.global_progress |
                             cplex.callbacks.Context.id.local_progress |
                             cplex.callbacks.Context.id.relaxation)
            cpx.parameters.mip.limits.nodes.set(1000)
            cpx.solve()
            # Make sure callback was actually invoked
            print(cb.calls, 'callback invocations')

            # We must have a completed cut loop for at least one node.
            # Note that we don't require this for all nodes: a node may
            # get pruned during the cut loop. In that case the callback
            # will not be invoked with "is after cut loop" = true.
            self.assertGreaterEqual(cb.calls, 1)
            nopen = len([filter(lambda x: not x, cb.nodes)])
            ncomplete = len([filter(lambda x: x, cb.nodes)])
            print('%d complete nodes, %d open' % (ncomplete, nopen))
            self.assertGreaterEqual(ncomplete, 1)

    def test_aflow30(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz')


if __name__ == '__main__':
    unittest.main()
