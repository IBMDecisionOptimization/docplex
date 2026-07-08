# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2019, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""Test case for pruning nodes from the Python generic callback.

The test is not very sophisticated. We assume that the real testing is
done with the C API. Here we just check that the pruning functions take
some effect.
We just prune any node at a certain depth D. We check that we never get
a node with depth larger than D and that we do not process more than 2^D
nodes. We assume that we do not perform a reopt before all nodes are
pruned.
"""
import unittest

import testutil
from cplextestcase import CplexTestCase
import cplex

RELDATADIR = '../../../examples/data'


class BranchOrCutCallback(object):
    """Generic callback that cuts all nodes at a certain depth."""

    def __init__(self, maxdepth):
        self._maxdepth = maxdepth
        self.pruned = 0

    def invoke(self, context):
        depth = context.get_long_info(cplex.callbacks.Context.info.node_depth)
        assert depth <= self._maxdepth

        if depth == self._maxdepth:
            context.prune_current_node()
            self.pruned += 1


class ExpertCallbackPrune(CplexTestCase):
    def _test_model(self, modelfile, maxdepth, context):
        '''Solve modelfile and cut nodes at depth maxdepth from context context.
        '''
        with cplex.Cplex() as cpx:
            if modelfile == 'markshare1':
                testutil.create_markshare1(cpx)
            else:
                cpx.read(modelfile)
            cb = BranchOrCutCallback(maxdepth)
            cpx.set_callback(cb, context)
            cpx.solve()
            # We assume there is no reopt
            limit = 1 if maxdepth == 0 else 2**(maxdepth + 1)
            assert cpx.solution.progress.get_num_nodes_processed() <= limit
            assert cb.pruned > 0


if __name__ == '__main__':
    # We want to test a number of combinations of models, maximum depths, and
    # callback types, so we generate the test cases automatically.
    for model, path in [('aflow30a', RELDATADIR + '/aflow30a.mps.gz'),
                        ('markshare1', 'markshare1'),
                        ('noswot', RELDATADIR + '/noswot.mps')]:
        for maxdepth in [0, 3]:
            for name, ctx in [('branch', cplex.callbacks.Context.id.branching),
                              ('cut', cplex.callbacks.Context.id.relaxation)]:
                setattr(ExpertCallbackPrune,
                        'test_%s_%d_%s' % (model, maxdepth, name),
                        lambda self: self._test_model(path, maxdepth, ctx))
    unittest.main()
