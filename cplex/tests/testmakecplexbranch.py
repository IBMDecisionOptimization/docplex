# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Test the make_cplex_branch function of the branch callback.

We perform several runs, in which we always create the branches that CPLEX
would have created anyway. Sometimes we do this by not doing anything,
sometimes we do it by calling make_cplex_branch() and sometimes we do it by
explicitly creating the branches that CPLEX would have created.

For each run we track the branching decisions that CPLEX performed at each
node. The expected result is that we always take the same path through the
tree.

ATTENTION: The test is a little shaky: Using node data or calling functions
           from inside the callback (make_branch(), make_cplex_branch()) may
           change the tick count and this in turn may change the path.
           We just cross our fingers that the tick changes triggered by the
           function calls are not enough to change the path.
"""
import sys
import threading
import testutil
import cplex
from cplex.callbacks import BranchCallback
from cplex.exceptions import CplexError

# Node limit applied during test.
NODELIMIT = 1000


class UserData:
    """A user object.

    The object just stores the sequence number of the node to which it is
    attached.

    The class also has static functions that keep track of which objects
    are attached to which nodes.
    """
    objects = dict()
    lock = threading.Lock()

    @staticmethod
    def reset():
        with UserData.lock:
            UserData.objects = dict()

    @staticmethod
    def addobject(key, obj):
        with UserData.lock:
            if key in UserData.objects:
                raise Exception("Duplicate key " + str(key))
            UserData.objects[key] = obj

    @staticmethod
    def getobject(key):
        ret = None
        with UserData.lock:
            if key in UserData.objects:
                ret = UserData.objects[key]
        return ret

    def __init__(self):
        self.seqnum = -1


class BranchCallbackBase(BranchCallback):
    """The base class for all callbacks defined here."""
    def __init__(self, env):
        super().__init__(env)
        self._lock = threading.Lock()
        self._history = dict()

    def _get_branchings(self):
        """Get the branchings that CPLEX plans to perform at this node.

        ALL callbacks invoke this function, even if they don't perform any
        custom branches. This is required because otherwise we can get
        different tick counts in different settings.

        The function also records the branching decisions so that we can
        later compare that two runs gave the same decisions.
        """
        try:
            self.get_branch(-1)
            raise AssertionError
        except CplexError:
            pass
        try:
            self.get_branch(self.get_num_branches())
            raise AssertionError
        except CplexError:
            pass
        branchcount = self.get_num_branches()
        branches = [self.get_branch(x) for x in range(branchcount)]
        threadno = self.get_thread_num()

        # Record the node number and the branching decisions CPLEX intended
        # to do here.
        with self._lock:
            if threadno not in self._history:
                self._history[threadno] = []
                self._history[threadno].append((self.get_node_ID(), branches))

        # Return the intended branches so that callbacks don't have to
        # query them again (which would modify the tick counts).
        return branches

    def get_history(self):
        return self._history


class EmptyCallback(BranchCallbackBase):
    """An empty branch callback that does nothing."""

    def __init__(self, env):
        super().__init__(env)

    def __call__(self):
        self._get_branchings()


class NullCallback(BranchCallbackBase):
    """A branch callback that invokes make_cplex_branch() without user
    object.
    """

    def __call__(self):
        branches = self._get_branchings()
        for which_branch in range(len(branches)):
            self.make_cplex_branch(which_branch)


class ObjectCallback(BranchCallbackBase):
    """A branch callback that invokes make_cplex_branch() with a user
    object."""

    def __call__(self):
        branches = self._get_branchings()
        nodeid = self.get_node_ID()
        if nodeid > 0:
            ndata = self.get_node_data()
            udata = UserData.getobject(nodeid)
            if ndata != udata:
                raise Exception("Invalid user data " + str(ndata) + " at node "
                                + str(nodeid) + " (expected " + str(udata) + ")")
            if ndata is not None and ndata.seqnum != nodeid:
                raise Exception("Invalid node data " + str(ndata.seqnum) +
                                " at node " + str(nodeid))
            if udata is not None and udata.seqnum != nodeid:
                raise Exception("Invalid user data " + str(udata.seqnum) +
                                " at node " + str(nodeid))
        for which_branch in range(len(branches)):
            udata = UserData()
            udata.seqnum = self.make_cplex_branch(which_branch, udata)
            UserData.addobject(udata.seqnum, udata)


class MixedCallback(BranchCallbackBase):
    """A branch callback that invokes make_cplex_branch() and
    make_branch() to replicate CPLEX branches.
    """

    def __init__(self, env):
        super().__init__(env)
        self.explicit_branch = 0

    def __call__(self):
        branches = self._get_branchings()
        nodeid = self.get_node_ID()
        if nodeid > 0:
            ndata = self.get_node_data()
            udata = UserData.getobject(nodeid)
            if ndata != udata:
                raise Exception("Invalid user data " + str(ndata) + " at node "
                                + str(nodeid) + " (expected " + str(udata) + ")")
            if ndata is not None and ndata.seqnum != nodeid:
                raise Exception("Invalid node data " + str(ndata.seqnum) +
                                " at node " + str(nodeid))
            if udata is not None and udata.seqnum != nodeid:
                raise Exception("Invalid user data " + str(udata.seqnum) +
                                " at node " + str(nodeid))
        branch = [self.get_branch(x)
                  for x in range(0, self.get_num_branches())]
        try:
            for which_branch in range(len(branches)):
                udata = UserData()
                if which_branch == self.explicit_branch:
                    # Each element in branches[] is a list (estimate,(vars,dirs,bnds))
                    b = branches[which_branch]
                    nums = self.make_branch(b[0], b[1], [], udata)
                    udata.seqnum = nums[0]
                else:
                    udata.seqnum = self.make_cplex_branch(which_branch, udata)
                UserData.addobject(udata.seqnum, udata)
        except Exception as er:
            print(str(er))
            raise


def solve(cpx, threads, cbtype, adjust = None):
    """Solve a problem."""
    # Setup parameters. Note that we _must_ run in deterministic mode as
    # we compare the outputs of different runs.
    cpx.parameters.threads.set(threads)
    cpx.parameters.mip.strategy.search.set(
        cpx.parameters.mip.strategy.search.values.traditional)
    cpx.parameters.mip.display.set(4)
    cpx.parameters.mip.limits.nodes.set(NODELIMIT)
    cpx.parameters.parallel.set(
        cpx.parameters.parallel.values.deterministic)
    # We set the display interval to 1 so in case of failure we can take
    # a good hard look at the offending logs.
    cpx.parameters.mip.interval.set(1)

    # Create the model.
    testutil.create_markshare1(cpx)

    UserData.reset()
    cb = cpx.register_callback(cbtype)
    if not adjust is None:
        adjust(cb)
    cpx.solve()
    return cb.get_history()


def checkhist(refhist, chkhist):
    """Compare a branching history to a reference.

    If you ever see an assertion failure here then make sure to read the
    ATTENTION at the top of this file before investigating things
    """
    lines = 0
    refthreads = set(key for key in refhist)
    chkthreads = set(key for key in chkhist)

    assert refthreads == chkthreads

    for threadno in sorted(refhist):
        for ref, chk in zip(refhist[threadno], chkhist[threadno]):
            assert ref[0] == chk[0] # node ids
            if ref[1] != chk[1]:
                print('%d: Different branch decisions at node %d' %
                      (threadno, ref[0]))
                print('REFERENCE: %s' % str(ref[1]))
                print('ACTUAL   : %s' % str(chk[1]))
            assert ref[1] == chk[1] # branching decisions at node

def main():
    """The main function."""
    with cplex.Cplex() as cpx:
        numcores = cpx.get_num_cores()

    if numcores > 2:
        step = 2
    else:
        step = 1

    for t in range(1, numcores, step):
        print("#### TEST ", t)

        # First solve: just solve the problem in default settings
        #              to create a reference log
        print("#### \tsolve 1 (no callbacks)")
        # This is actually useless, so we skip it: Running with an empty
        # callback is definitely not the same as running without callback.
        
        # Second solve: solve with an empty branch callback
        print("#### \tsolve 2 (empty callback)")
        with cplex.Cplex() as cpx:
            hist2 = solve(cpx, t, EmptyCallback)
            assert t == len(hist2), \
                "expecting {0} threads, was {1}".format(t, len(hist2))

        # Third solve: solve with a callback that duplicates CPLEX branches
        #              but does not add user data.
        print("#### \tsolve 3 (null callback)")
        with cplex.Cplex() as cpx:
            hist3 = solve(cpx, t, NullCallback)
        checkhist(hist2, hist3)

        # Fourth solve: solve with a callback that duplicates CPLEX branches
        #               and adds user data.
        print("#### \tsolve 4 (object callback)")
        with cplex.Cplex() as cpx:
            hist4 = solve(cpx, t, ObjectCallback)
        checkhist(hist3, hist4)

        # We need this since lambdas cannot contain assignments
        def set_explicit(cb, which): cb.explicit_branch = which

        # Fourth solve: solve with a callback that duplicates CPLEX branches
        #               and adds user data. CPLEX branches are duplicate by a
        #               mix of make_branch() and make_cplex_branch().
        print("#### \tsolve 5 (mixed callback, explicit branch 0)")
        with cplex.Cplex() as cpx:
            hist5 = solve(cpx, t, MixedCallback, lambda x: set_explicit(x, 0))
        # Too bad: The tick changes implied by calling more functions in the
        #          callback are just enought to make this fail in
        #          multi-threaded runs
        if t == 1:
            checkhist(hist4, hist5)

        # Sixth solve: solve with a callback that duplicates CPLEX branches
        #              and adds user data. CPLEX branches are duplicate by a
        #              mix of make_branch() and make_cplex_branch().
        print("#### \tsolve 6 (mixed callback, explicit branch 1)")
        with cplex.Cplex() as cpx:
            hist6 = solve(cpx, t, MixedCallback, lambda x: set_explicit(x, 1))
        checkhist(hist5, hist6)

    print("Test passed")


if __name__ == "__main__":
    main()
