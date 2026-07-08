# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55
# Copyright IBM Corporation 2019, 2026. All Rights Reserved.
# 
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Test the get_current_node_depth() function of callbacks."""
import cplex
import sys
import threading
import unittest
from cplextestcase import CplexTestCase
from cplex.exceptions.error_codes import CPXERR_UNSUPPORTED_OPERATION

RELDATADIR = '../../../examples/data'

class DepthCheck:
    """Check that the depths of nodes are as expected.

    The depth of the root is 0 and the depth of every other node is
    1 plus the depth of its parent. Expected depths values are registered
    from a branch callback and checked from all other callbacks."""
    def __init__(self):
        pass
    def setup(self, lock, depth_map):
        # Not in the constructor due to weird callback registry protocol
        self._lock = lock
        self._depth_map = depth_map
    def set_depth(self, node, depth):
        with self._lock:
            assert node not in self._depth_map
            self._depth_map[node] = depth
    def check_depth(self, cb):
        nodeid = cb.get_node_ID()
        depth = cb.get_current_node_depth()
        assert depth == self._depth_map[nodeid]

class MyUserCutCallback(cplex.callbacks.UserCutCallback, DepthCheck):

    def __call__(self):
        self.check_depth(self)


class MyLazyConstraintCallback(cplex.callbacks.LazyConstraintCallback, DepthCheck):

    def __call__(self):
        self.check_depth(self)


class MyHeuristicCallback(cplex.callbacks.HeuristicCallback, DepthCheck):

    def __call__(self):
        self.check_depth(self)


class MySolveCallback(cplex.callbacks.SolveCallback, DepthCheck):

    def __call__(self):
        self.check_depth(self)


class MyIncumbentCallback(cplex.callbacks.IncumbentCallback, DepthCheck):

    def __call__(self):
        self.check_depth(self)


class MyBranchCallback(cplex.callbacks.BranchCallback, DepthCheck):

    def __call__(self):
        self.check_depth(self)
        for i in range(self.get_num_branches()):
            child = self.make_cplex_branch(i)
            self.set_depth(child, self.get_current_node_depth() + 1)


class MyNodeCallback(cplex.callbacks.NodeCallback, DepthCheck):

    def __call__(self):
        try:
            # For the node callback the function must raise an exception
            # since the node callback has no current node
            self.get_current_node_depth()
            assert False
        except cplex.exceptions.CplexSolverError as e:
            assert e.args[0] == 'Not in a node context'
            assert e.args[1] is None # env
            assert e.args[2] == CPXERR_UNSUPPORTED_OPERATION

class TestNodeDepth(CplexTestCase):
    def _test_model(self, model):
        global depth_map
        with cplex.Cplex(model) as cpx:
            depth_map = { 0 : 0 } # root node
            lock = threading.Lock()
            cpx.parameters.threads.set(cpx.get_num_cores())
            cpx.register_callback(MyUserCutCallback).setup(lock, depth_map)
            cpx.register_callback(MyLazyConstraintCallback).setup(lock, depth_map)
            cpx.register_callback(MyHeuristicCallback).setup(lock, depth_map)
            cpx.register_callback(MySolveCallback).setup(lock, depth_map)
            cpx.register_callback(MyIncumbentCallback).setup(lock, depth_map)
            cpx.register_callback(MyBranchCallback).setup(lock, depth_map)
            cpx.register_callback(MyNodeCallback).setup(lock, depth_map)
            cpx.solve()

            # We assume that 100 nodes are enough to perform reasonable
            # tests in the callbacks
            self.assertGreater(len(depth_map), 100)
        print("Test passed")
    def test_aflow30a(self): self._test_model(RELDATADIR + "/aflow30a.mps.gz")


if __name__ == "__main__":
    unittest.main()
