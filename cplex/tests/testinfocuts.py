# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
# 
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Test code adapted from admipex5.py:

Solves the MIPLIB 3.0 model noswot.mps:
 - with infocallbacks
 - by adding cuts via a user cut callback during the branch-and-cut process.
Tests that from mipinfo callbacks and from cutcallbacks querying the number
of active lift-and-project cuts, user cuts and lazy constraints, is
actually ok and does not produce any error.

You can run this example at the command line by

  python testinfocuts.py

or within the python interpreter by

  >>> import testinfocuts
"""
import sys
import traceback

import cplex
from cplex.callbacks import UserCutCallback, LazyConstraintCallback, MIPInfoCallback


class MyInfo(MIPInfoCallback):

    def __init__(self, env):
        super().__init__(env)
        self.printinfo = 0
        
    def __call__(self):
        try:
            for which in self.cut_type:
                assert self.get_num_cuts(which) >= 0

            nmir      = self.get_num_cuts(self.cut_type.MIR)
            ngom      = self.get_num_cuts(self.cut_type.fractional)
            nliftproj = self.get_num_cuts(self.cut_type.lift_and_project)
            nuser     = self.get_num_cuts(self.cut_type.user)
            nlazy     = self.get_num_cuts(self.cut_type.table)
            npoolcuts = self.get_num_cuts(self.cut_type.solution_pool)
            if self.printinfo == 1:
                print(("nmir      = %d" % nmir))
                print(("ngom      = %d" % ngom))
                print(("nliftproj = %d" % nliftproj))

            print('MyInfo: nuser=%d, nlazy=%d, npoolcuts=%d' %
                  (nuser, nlazy, npoolcuts))
            assert nuser == 0
            assert nlazy == 0
            assert npoolcuts == 0
        except:
            traceback.print_exc()
            raise


#  Add the following valid cuts for the noswot model via cut callback:
#
#  cut1: X21 - X22 <= 0
#  cut2: X22 - X23 <= 0
#  cut3: X23 - X24 <= 0
#  cut4: 2.08 X11 + 2.98 X21 + 3.47 X31 + 2.24 X41 + 2.08 X51 
#      + 0.25 W11 + 0.25 W21 + 0.25 W31 + 0.25 W41 + 0.25 W51
#        <= 20.25
#  cut5: 2.08 X12 + 2.98 X22 + 3.47 X32 + 2.24 X42 + 2.08 X52 
#      + 0.25 W12 + 0.25 W22 + 0.25 W32 + 0.25 W42 + 0.25 W52
#        <= 20.25
#  cut6: 2.08 X13 + 2.98 X23 + 3.4722 X33 + 2.24 X43 + 2.08 X53
#      + 0.25 W13 + 0.25 W23 + 0.25 W33 + 0.25 W43 + 0.25 W53
#        <= 20.25
#  cut7: 2.08 X14 + 2.98 X24 + 3.47 X34 + 2.24 X44 + 2.08 X54 
#      + 0.25 W14 + 0.25 W24 + 0.25 W34 + 0.25 W44 + 0.25 W54
#        <= 20.25
#  cut8: 2.08 X15 + 2.98 X25 + 3.47 X35 + 2.24 X45 + 2.08 X55 
#      + 0.25 W15 + 0.25 W25 + 0.25 W35 + 0.25 W45 + 0.25 W55
#        <= 16.25
#

def initcuts():
    lhs = []
    rhs = []

    lhs = lhs + [cplex.SparsePair(ind = ["X21", "X22"], val = [1.0, -1.])]
    rhs = rhs + [0.0]
        
    lhs = lhs + [cplex.SparsePair(ind = ["X22", "X23"], val = [1.0, -1.])]
    rhs = rhs + [0.0]
        
    lhs = lhs + [cplex.SparsePair(ind = ["X23", "X24"], val = [1.0, -1.])]
    rhs = rhs + [0.0]
        
    lhs = lhs + [cplex.SparsePair(ind = ["X11", "X21", "X31", "X41", "X51",
                                         "W11", "W21", "W31", "W41", "W51"],
                                  val = [2.08, 2.98, 3.47, 2.24, 2.08,
                                         0.25, 0.25, 0.25, 0.25, 0.25])]
    rhs = rhs + [20.25]

    lhs = lhs + [cplex.SparsePair(ind = ["X12", "X22", "X32", "X42", "X52",
                                         "W12", "W22", "W32", "W42", "W52"],
                                  val = [2.08, 2.98, 3.47, 2.24, 2.08,
                                         0.25, 0.25, 0.25, 0.25, 0.25])]
    rhs = rhs + [20.25]
        
    lhs = lhs + [cplex.SparsePair(ind = ["X13", "X23", "X33", "X43", "X53",
                                         "W13", "W23", "W33", "W43", "W53"],
                                  val = [2.08, 2.98, 3.4722, 2.24, 2.08,
                                         0.25, + 0.25, 0.25, 0.25, 0.25])]
    rhs = rhs + [20.25]

    lhs = lhs + [cplex.SparsePair(ind = ["X14", "X24", "X34", "X44", "X54",
                                         "W14", "W24", "W34", "W44", "W54"],
                                  val = [2.08, 2.98, 3.47, 2.24, 2.08,
                                         0.25, 0.25, 0.25, 0.25, 0.25])]
    rhs = rhs + [20.25]
        
    lhs = lhs + [cplex.SparsePair(ind = ["X15", "X25", "X35", "X45", "X55",
                                         "W15", "W25", "W35", "W45", "W55"],
                                  val = [2.08, 2.98, 3.47, 2.24, 2.08,
                                         0.25, 0.25, 0.25, 0.25, 0.25])]
    rhs = rhs + [16.25]

    return (lhs, rhs)


class MyCut(UserCutCallback):
    def __init__(self, env):
        super().__init__(env)
        self.lhs, self.rhs = initcuts()
        self.num_cb_usercuts = 0
        self.printinfo = 0

    def __call__(self):
        try:
            nmir      = self.get_num_cuts(self.cut_type.MIR)
            ngom      = self.get_num_cuts(self.cut_type.fractional)
            nliftproj = self.get_num_cuts(self.cut_type.lift_and_project)
            nuser     = self.get_num_cuts(self.cut_type.user)
            nlazy     = self.get_num_cuts(self.cut_type.table)
            npoolcuts = self.get_num_cuts(self.cut_type.solution_pool)
            if self.printinfo == 1:
                print(("nmir      = %d" % nmir))
                print(("ngom      = %d" % ngom))
                print(("nliftproj = %d" % nliftproj))

            print('MyCut: nuser=%d (%d), nlazy=%d, npoolcuts=%d' %
                  (nuser, self.num_cb_usercuts, nlazy, npoolcuts))
            # Lazy constraints added via callback are counted as user cuts
            assert nuser >= self.num_cb_usercuts
            assert nlazy == 0
            assert npoolcuts == 0

            # loop through our list of cuts and check whether they are violated
            lhs = self.lhs
            rhs = self.rhs
            nCuts = len(rhs)
            for i in range(nCuts):
                # calculate activity of cut
                act = 0
                cutlen = len(lhs[i].ind)
                for k in range(cutlen):
                    j = lhs[i].ind[k]
                    a = lhs[i].val[k]
                    act += a*self.get_values(j)

                # check if cut is violated
                if act > rhs[i] + 1e-6:
                    self.add(cut = lhs[i], sense = "L", rhs = rhs[i],
                             use = self.use_cut.force)
                    self.num_cb_usercuts = self.num_cb_usercuts + 1
        except:
            traceback.print_exc()
            raise


class MyLazy(LazyConstraintCallback):
    def __init__(self, env):
        super().__init__(env)
        self.lhs, self.rhs = initcuts()
        self.num_cb_usercuts = 0

    def __call__(self):
        try:
            # loop through our list of cuts and check whether they are violated
            lhs = self.lhs
            rhs = self.rhs
            nCuts = len(rhs)
            for i in range(nCuts):
                # calculate activity of cut
                act = 0
                cutlen = len(lhs[i].ind)
                for k in range(cutlen):
                    j = lhs[i].ind[k]
                    a = lhs[i].val[k]
                    act += a*self.get_values(j)

                # check if cut is violated
                if act > rhs[i] + 1e-6:
                    self.add(constraint = lhs[i], sense = "L", rhs = rhs[i],
                             use = self.use_constraint.force)
                    self.num_cb_usercuts = self.num_cb_usercuts + 1
        except:
            traceback.print_exc()
            raise


def solve_and_report(c, usercutcb, lazyconscb, useuserandlazy, gomagg, landpagg):
    # set desired level of Gomory and LiftProj cuts
    c.parameters.mip.cuts.gomory.set(gomagg)
    c.parameters.mip.cuts.liftproj.set(landpagg)

    # solve problem
    c.solve()

    # get expected number of user cuts
    num_cb_usercuts = 0
    if useuserandlazy == 1:
        num_cb_usercuts = usercutcb.num_cb_usercuts + lazyconscb.num_cb_usercuts

    nmir      = c.solution.MIP.get_num_cuts(c.solution.MIP.cut_type.MIR)
    ngom      = c.solution.MIP.get_num_cuts(c.solution.MIP.cut_type.fractional)
    nliftproj = c.solution.MIP.get_num_cuts(c.solution.MIP.cut_type.lift_and_project)
    nuser     = c.solution.MIP.get_num_cuts(c.solution.MIP.cut_type.user)
    nlazy     = c.solution.MIP.get_num_cuts(c.solution.MIP.cut_type.table)
    npoolcuts = c.solution.MIP.get_num_cuts(c.solution.MIP.cut_type.solution_pool)
    
    print(("nmir       = %d" % nmir))
    print(("ngom       = %d" % ngom))
    print(("nliftproj  = %d" % nliftproj))
    print(("nuser      = %d" % nuser))
    print(("nlazy      = %d" % nlazy))
    print(("npoolcuts  = %d" % npoolcuts))

    print(("num_cb_usercuts  = %d" % num_cb_usercuts))
        
    # Lazy constraints added via callback are counted as user cuts 
    assert nuser == num_cb_usercuts
    assert nlazy == 0
    assert npoolcuts == 0

    if useuserandlazy == 1: 
        assert nuser > 0

    if gomagg == -1:
        assert ngom == 0
    if landpagg == -1:
        assert nliftproj == 0

    # set default level of Gomory and LiftProj cuts
    c.parameters.mip.cuts.gomory.set(0)
    c.parameters.mip.cuts.liftproj.set(0)
        
    
def testinfocuts():

    c = cplex.Cplex("../../../examples/data/noswot.mps")

    # sys.stdout is the default output stream for log and results
    # so these lines may be omitted
    c.set_log_stream(sys.stdout)
    c.set_results_stream(sys.stdout)

    # set node log interval to 1000
    c.parameters.mip.interval.set(1000)

    # set mip node limit to 1000
    c.parameters.mip.limits.nodes.set(1000)

    # the problem will be solved several times, so turn off advanced start
    c.parameters.advance.set(0)

    # RTC-16136
    # Test that new parameter CPX_PARAM_CALCQCPDUALS 
    # can be accessed and modified.
    oldqcpduals = c.parameters.preprocessing.qcpduals.get()
    c.parameters.preprocessing.qcpduals.set(2)
    newqcpduals = c.parameters.preprocessing.qcpduals.get()
    assert newqcpduals == 2
    c.parameters.preprocessing.qcpduals.set(oldqcpduals)

    # solve the problem with infocallback
    c.parameters.mip.strategy.search.set(c.parameters.mip.strategy.search.values.dynamic)
    c.parameters.threads.set(0)
    c.register_callback(MyInfo)
    solve_and_report(c, 0, 0, 0, -1, -1)
    solve_and_report(c, 0, 0, 0, -1, 3)
    solve_and_report(c, 0, 0, 0, 0, -1)
    solve_and_report(c, 0, 0, 0, 0, 3)
    c.unregister_callback(MyInfo)

    # solve the problem with cutcallback
    c.parameters.mip.strategy.search.set(c.parameters.mip.strategy.search.values.traditional)
    c.parameters.threads.set(1)
    usercutcb = c.register_callback(MyCut)
    lazyconscb = c.register_callback(MyLazy)
    solve_and_report(c, usercutcb, lazyconscb, 1, 0, 2)
    c.unregister_callback(MyCut)
    c.unregister_callback(MyLazy)

    print ("Test complete")

    
if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: testinfocuts.py ")
        sys.exit(-1)
    testinfocuts()
