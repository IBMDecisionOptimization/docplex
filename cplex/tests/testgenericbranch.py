# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2019, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""Test that extends the genericbranch.py example.

Compared to genericbranch.py, this test contains a lot of additional code
that tests that branching decisions are actually respected when solving
the node LPs. The additional code also tests the various overloads of the
make_branch() method.
"""
from collections import namedtuple
import math
import sys
import threading
import traceback
import unittest

import cplex
from cplex import SparsePair

from cplextestcase import CplexTestCase

NBRANCHTYPES = 5  # Number of different types of branches we create.
BranchInfo = namedtuple("BranchInfo", ["var", "bound", "islb"])


class BranchCallback:
    """Generic callback that implements most infeasible branching."""

    def __init__(self, x):
        self._x = x
        self.calls = 0
        self.branches = 0
        self._info_map = dict()
        self._parent_map = dict()
        self._mtx = threading.Lock()

    def invoke(self, context):
        try:
            self.calls += 1

            depth = context.get_long_info(
                cplex.callbacks.Context.info.node_depth)
            assert depth <= 1000

            status = context.get_relaxation_status()
            assert status == context.solution_status.optimal, \
                "unexpected status {0}".format(status)

            obj = context.get_relaxation_objective()
            thread_id = context.get_int_info(
                cplex.callbacks.Context.info.thread_id)
            eps = 1e-6

            # Read current relaxation into a map so that we can easily test
            # whether it satisfies all branching decisions.
            var2val = {
                j: v for j, v
                in zip(self._x, context.get_relaxation_point(self._x))
            }
            this_node_id = context.get_long_info(
                cplex.callbacks.Context.info.node_uid)

            with self._mtx:
                if this_node_id not in self._info_map:
                    # Not in the node map: Must be a root node.
                    depth = context.get_long_info(
                        cplex.callbacks.Context.info.node_depth)
                    assert depth == 0, ('unexpected depth %d for node %d' %
                                        (depth, this_node_id))
                else:
                    # Make sure that the values in the current relaxation
                    # satisfy all branching decisions that lead to this
                    # node.
                    node_id = this_node_id
                    while node_id in self._info_map:
                        for b in self._info_map[node_id]:
                            val = var2val[b.var]
                            if b.islb:
                                assert val >= b.bound - eps, \
                                    '%f < %f (%g)' % (val, b.bound - eps,
                                                      abs(val - (b.bound - eps)))
                            else:
                                assert val <= b.bound + eps, \
                                    '%f > %f (%g)' % (val, b.bound + eps,
                                                      abs(val - (b.bound + eps)))
                        node_id = self._parent_map[node_id]

            # Also collect the second best variable so that we
            # can branch on more than one variable
            max_var2 = -2
            max_frac2 = 0.0
            max_val2 = 0.0

            # The Node LP was solved to optimality. Grab the current
            # relaxation and find the most fractional variable.
            max_var = -1
            max_frac = 0.0
            max_val = 0.0
            for j, v in zip(self._x, context.get_relaxation_point(self._x)):
                intval = round(v)
                frac = abs(intval - v)

                if frac > max_frac:
                    max_frac2 = max_frac
                    max_var2 = max_var
                    max_val2 = max_val

                    max_frac = frac
                    max_var = j
                    max_val = v
                elif frac > max_frac2:
                    max_frac2 = frac
                    max_var2 = j
                    max_val2 = v

            # Always branch if max_frac so that we can easily
            # track the whole tree.
            if max_frac:
                up2 = math.ceil(max_val2) if max_frac2 else 0.0
                down2 = math.floor(max_val2) if max_frac2 else 0.0
                branch_var2 = max_var2 if max_frac2 else -1

                up = math.ceil(max_val)
                down = math.floor(max_val)
                branch_var = max_var

                which = thread_id % (NBRANCHTYPES if max_frac2 else 2)
                if which == 0:
                    # Branch on single variable
                    # Create UP branch (branch_var >= up)
                    up_child = context.make_branch(obj, [(branch_var, 'L', up)])

                    # Create DOWN branch (branch_var <= down)
                    down_child = context.make_branch(
                        obj,
                        [(branch_var, 'U', down)]
                    )

                    with self._mtx:
                        self._info_map[up_child] = [
                            BranchInfo(branch_var, up, True)
                        ]
                        self._parent_map[up_child] = this_node_id
                        self._info_map[down_child] = [
                            BranchInfo(branch_var, down, False)
                        ]
                        self._parent_map[down_child] = this_node_id
                elif which == 1:
                    # Branch on single variable as constraint.
                    up_child = context.make_branch(
                        obj,
                        None,
                        [(SparsePair(ind=[branch_var], val=[1.0]), 'G', up)]
                    )
                    down_child = context.make_branch(
                        obj,
                        None,
                        [(SparsePair(ind=[branch_var], val=[1.0]), 'L', down)]
                    )
                    with self._mtx:
                        self._info_map[up_child] = [
                            BranchInfo(branch_var, up, True)
                        ]
                        self._parent_map[up_child] = this_node_id
                        self._info_map[down_child] = [
                            BranchInfo(branch_var, down, False)
                        ]
                        self._parent_map[down_child] = this_node_id
                elif which == 2:
                    # Branch on two variables simultaneously
                    up_child = context.make_branch(
                        obj,
                        [(branch_var, 'L', up), (branch_var2, 'L', up2)]
                    )
                    down_child = context.make_branch(
                        obj,
                        [(branch_var, 'U', down), (branch_var2, 'U', down2)]
                    )
                    with self._mtx:
                        self._info_map[up_child] = [
                            BranchInfo(branch_var, up, True),
                            BranchInfo(branch_var2, up2, True)
                        ]
                        self._parent_map[up_child] = this_node_id
                        self._info_map[down_child] = [
                            BranchInfo(branch_var, down, False),
                            BranchInfo(branch_var2, down2, False)
                        ]
                        self._parent_map[down_child] = this_node_id
                elif which == 3:
                    # Branch on two variables as constraint.
                    up_child = context.make_branch(
                        obj,
                        None,
                        [(SparsePair(ind=[branch_var], val=[1.0]), 'G', up),
                         (SparsePair(ind=[branch_var2], val=[1.0]), 'G', up2)]
                    )
                    down_child = context.make_branch(
                        obj,
                        None,
                        [(SparsePair(ind=[branch_var], val=[1.0]), 'L', down),
                         (SparsePair(ind=[branch_var2], val=[1.0]), 'L', down2)]
                    )
                    with self._mtx:
                        self._info_map[up_child] = [
                            BranchInfo(branch_var, up, True),
                            BranchInfo(branch_var2, up2, True)
                        ]
                        self._parent_map[up_child] = this_node_id
                        self._info_map[down_child] = [
                            BranchInfo(branch_var, down, False),
                            BranchInfo(branch_var2, down2, False)
                        ]
                        self._parent_map[down_child] = this_node_id
                else:
                    assert which == 4
                    # Branch on two variables, one as bound change, one as
                    # constraint.
                    up_child = context.make_branch(
                        obj,
                        [(branch_var, 'L', up)],
                        [(SparsePair(ind=[branch_var2], val=[1.0]), 'G', up2)]
                    )
                    down_child = context.make_branch(
                        obj,
                        [(branch_var, 'U', down)],
                        [(SparsePair(ind=[branch_var2], val=[1.0]), 'L', down2)]
                    )
                    with self._mtx:
                        self._info_map[up_child] = [
                            BranchInfo(branch_var, up, True),
                            BranchInfo(branch_var2, up2, True)
                        ]
                        self._parent_map[up_child] = this_node_id
                        self._info_map[down_child] = [
                            BranchInfo(branch_var, down, False),
                            BranchInfo(branch_var2, down2, False)
                        ]
                        self._parent_map[down_child] = this_node_id

                self.branches += 1

        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


class TestGenericBranch(CplexTestCase):

    def _test_model(self, model):
        with self._newCplex(model) as cpx:
            ctype = cpx.variables.get_types()
            # Create a callback and pass as argument the indices of all
            # non-continuous variables.
            cb = BranchCallback([i for i, c in enumerate(ctype) if c != 'C'])

            # Register the callback with CPLEX and ask CPLEX to invoke it
            # only in the branching context.
            cpx.set_callback(cb, cplex.callbacks.Context.id.branching)

            # Limit the number of nodes.
            # The branching strategy implemented here is not smart so
            # solving even a simple MIP may turn out to take a long time.
            cpx.parameters.mip.limits.nodes.set(1000)

            # Use as many threads as we have branching decision types.
            # This is to make sure we use all those decisions and thus
            # cover all the respective code.
            threads = cpx.get_num_cores()
            if threads < NBRANCHTYPES:
                threads = NBRANCHTYPES
                cpx.parameters.threads.set(threads)

            # Solve the model and report some statistics.
            cpx.solve()

            self.assertGreater(cb.calls, 0)
            self.assertGreater(cb.branches, 0)

    def test_noswot(self):
        self._test_model(self._getResource("examples/data/noswot.mps"))

    def test_aflow30a(self):
        self._test_model("../../data/aflow30a.mps.gz")


if __name__ == '__main__':
    unittest.main()
