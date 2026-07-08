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
Tests the unregister_callback method.

No command line arguments are required.
"""
import unittest
import os
from cplextestcase import CplexTestCase
from cplex.callbacks import BranchCallback
from cplex.callbacks import LazyConstraintCallback
from cplex.callbacks import UserCutCallback
from cplex.callbacks import HeuristicCallback
from cplex.callbacks import SolveCallback
from cplex.callbacks import NodeCallback
from cplex.callbacks import IncumbentCallback

class NoOp:

    called_once = False

    def __call__(self):
        self.called_once = True
        self.abort()


class BranchNoOp(NoOp, BranchCallback):
    pass


class LazyNoOp(NoOp, LazyConstraintCallback):
    pass


class CutNoOp(NoOp, UserCutCallback):
    pass


class HeuristicNoOp(NoOp, HeuristicCallback):
    pass


class SolveNoOp(NoOp, SolveCallback):
    pass


class NodeNoOp(NoOp, NodeCallback):
    pass


class IncumbentNoOp(NoOp, IncumbentCallback):
    pass


# TODO: There are more callback classes we could be testing:
#       TuningCallback
#       CrossoverCallback
#       etc.

class RemoveCallbackTests(CplexTestCase):

    def testBranchCallback(self):
        self._testRemoval(BranchNoOp)

    def testLazyConstraintCallback(self):
        self._testRemoval(LazyNoOp)

    def testUserCutCallback(self):
        self._testRemoval(CutNoOp)

    def testHeuristicCallback(self):
        self._testRemoval(HeuristicNoOp)

    def testSolveCallback(self):
        self._testRemoval(SolveNoOp)

    def testNodeCallback(self):
        self._testRemoval(NodeNoOp)

    def testIncumbentCallback(self):
        self._testRemoval(IncumbentNoOp)

    def _testRemoval(self, cbk):
        with self._newCplex() as c:
            c.read(self._getResource("examples/data/noswot.mps"))
            c.parameters.mip.limits.nodes.set(1000)
            c.parameters.advance.set(0)
            # first solve without callback
            with open("out0", "w") as out:
                c.set_warning_stream(out)
                c.solve()
            # second solve with callback
            with open("out1", "w") as out:
                cb = c.register_callback(cbk)
                c.set_warning_stream(out)
                c.solve()
                # make sure the callback was invoked
                self.assertTrue(cb.called_once)
            # third solve without callback
            unreg = c.unregister_callback(cbk)
            # To test repeatedly registering/unregistering a callback, do
            # it again (see RTC-36815).
            cb = c.register_callback(cbk)
            unreg = c.unregister_callback(cbk)
            with open("out2", "w") as out:
                c.set_warning_stream(out)
                c.solve()

        with open("out0", "r") as f:
            # should be no warnings from the first solve call
            self.assertEqual(f.read(), "")
        with open("out1", "r") as f:
            # should be warning about disabling dynamic search
            self.assertEqual(
                f.read(),
                "Warning: Control callbacks may disable some MIP features.\n")
        with open("out2", "r") as f:
            # should be no warnings from the last solve call
            self.assertEqual(f.read(), "")

        self._failSafeDelete("out0")
        self._failSafeDelete("out1")
        self._failSafeDelete("out2")


def main():
    unittest.main()

if __name__ == '__main__':
    main()
