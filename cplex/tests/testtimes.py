# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
# 
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Test the time query functions."""
from math import fabs
import cplex as CPX
import cplex.callbacks as CPX_CB
import sys
import threading
import testutil

# Tolerance (in seconds) we allow for deviation of wallclock time
TIMTOL = 2
# Tolerance (in ticks) we allow for deviation of deterministic time
DETTOL = 1

DETLIM = 0
TIMLIM = 0

# Time at which the solution process started
class Times:
    startdet = 0  # deterministic time
    starttim = 0  # wallclock time
    checks = 0 # How many checks did we perform?
    lock = threading.Lock()

# Node callback to test the time query functions that can be
# invoked from a callback.
class MyNodeCallback(CPX_CB.NodeCallback):

    def __call__(self):
        # Get current time stamp.
        det = self.get_dettime()
        tim = self.get_time()

        # Check that time is increasing.
        # Note that we must use thread-local data to store the last
        # timestamp since Python has no mechanism for cloning callbacks
        # for each thread.
        if not hasattr(self, "lastdet"):
            self.lastdet = threading.local()
        if hasattr(self.lastdet, "val"):
            if self.lastdet.val > det:
                print("lastdet=", self.lastdet.val, " det=", det, file=sys.stderr)
                raise Exception("Deterministic time is not increasing")
        self.lastdet.val = det
        if not hasattr(self, "lasttim"):
            self.lasttim = threading.local()
        if hasattr(self.lasttim, "val"):
            if self.lasttim.val > tim:
                print("lasttim=", self.lasttim.val, " tim=", tim, file=sys.stderr)
                raise Exception("Wallclock time is not increasing")
        self.lasttim.val = tim


        Times.lock.acquire()
        
        # Check that time is greater than starting time
        if det < Times.startdet:
            raise Exception("Invalid deterministic time value")
        if self.get_start_dettime() < Times.startdet:
            raise Exception("Invalid deterministic start time")
        if tim < Times.starttim:
            raise Exception("Invalid wallclock time value")
        if self.get_start_time() < Times.starttim:
            raise Exception("Invalid wallclock start time")
        Times.checks += 1

        # Check that time limits are returned correctly
        detlim = self.get_end_dettime()
        timlim = self.get_end_time()
        expected = Times.starttim + TIMLIM
        if fabs(timlim - expected) > TIMTOL:
            raise Exception("Callback.get_end_time: got " + str(timlim) +
                            ", expected " + str(expected) +
                            " (delta=" + str(fabs(timlim - expected)) +
                            ", tolerance=" + str(TIMTOL) + ")")
        expected = Times.startdet + DETLIM
        if fabs(detlim - expected) > DETTOL:
            print("startdet=",Times.startdet,", DETLIM=",DETLIM,", detlim=",detlim, file=sys.stderr)
            raise Exception("Callback.get_end_dettime: got " + str(detlim) +
                            ", expected " + str(expected) +
                            " (delta=" + str(fabs(detlim - expected)) +
                            ", tolerance=" + str(DETTOL) + ")")
        
        Times.lock.release()



if __name__ == "__main__":
    detlimhits = 0
    timlimhits = 0

    # Perform two solves for 10 nodes to figure out a good
    # time limit for tests.
    print("#### Calibrating")
    TESTDETLIM = 0
    TESTTIMLIM = 0
    NUMPROCS = 1
    for t in [1,2]:
        c = CPX.Cplex()
        NUMPROCS = c.get_num_cores()
        testutil.create_markshare1(c)

        c.parameters.threads.set(t)
        c.parameters.mip.limits.nodes.set(10)
        start = c.get_time()
        detstart = c.get_dettime()
        c.solve()
        delta = c.get_time() - start
        detdelta = c.get_dettime() - detstart
        
        if delta > TESTTIMLIM:
            TESTTIMLIM = delta
        if detdelta > TESTDETLIM:
            TESTDETLIM = detdelta
    TESTTIMLIM += 10 # Run for at least ten seconds (too small will fail on svcplexdev2 and sles10)
    TESTDETLIM += 200 # Run for at least two hundred deterministic ticks
    print("#### Using TESTDETLIM = " + str(TESTDETLIM))
    print("#### Using TESTTIMLIM = " + str(TESTTIMLIM))

    # Run with two different time limit configurations so that we hit all
    # types of time limits. The maximum non-infinite value for deterministic
    # time limits is a little larger than 1e12 (2^64/2^20), so we just use
    # 1e12 for the cases that are supposed to hit the wallclock time limit.
    for (detlim, timlim) in ((1e12, TESTTIMLIM), (TESTDETLIM, 1e20)):
        for t in (0, 1, 2): # 0: automatic, 1: sequential, N: use N threads
            # We purposely do not test CPU time (see RTC-18648).  Therefore,
            # we cannot use c.parameters.clocktype.values.CPU (obviously) nor
            # c.parameters.clocktype.values.auto (which will select CPU time
            # when only using one thread).
            for clock in (c.parameters.clocktype.values.wall,):
                for mode in (c.parameters.parallel.values.opportunistic,
                             c.parameters.parallel.values.auto,
                             c.parameters.parallel.values.deterministic):
                    DETLIM = detlim
                    TIMLIM = timlim
                    print("#### Testing clock=",clock,", threads=",t,", mode=",mode,", detlim=",DETLIM,", timlim=",TIMLIM)
                    
                    c = CPX.Cplex()
                    testutil.create_markshare1(c)

                    c.parameters.threads.set(t)
                    c.parameters.parallel.set(mode)
                    c.parameters.clocktype.set(clock)
                    c.parameters.timelimit.set(TIMLIM)
                    c.parameters.dettimelimit.set(DETLIM)
                    c.register_callback(MyNodeCallback)
                
                    Times.startdet = c.get_dettime()
                    Times.starttim = c.get_time()
                    Times.checks = 0
                    c.solve()

                    # Record timestamps right after solve.
                    det = c.get_dettime()
                    tim = c.get_time()

                    # We did a non-trivial solve so any kind of time must
                    # have increased.
                    if det <= Times.startdet:
                        raise Exception("Invalid deterministic time " +
                                        str(det) +
                                        " expected something >= " +
                                        str(time.startdet))
                    if tim <= Times.starttim:
                        raise Exception("Invalid wallclock time" +
                                        str(tim) +
                                        "expected somthing >= " +
                                        str(time.starttim))
                    if Times.checks < 1:
                        raise Exception("Callback was never invoked")

                    # We are supposed to either hit deterministic or
                    # wallclock time limit.
                    detstat = [ c.solution.status.MIP_dettime_limit_feasible,
                                c.solution.status.MIP_dettime_limit_infeasible ]
                    timstat = [ c.solution.status.MIP_time_limit_feasible,
                                c.solution.status.MIP_time_limit_infeasible ]
                    if c.solution.get_status() in detstat:
                        detlimhits += 1
                        expected = Times.startdet + DETLIM
                        if expected > det:
                            raise Exception("Deterministic time limit expired too early (" +
                                            str(expected) + " vs. " +
                                            str(det) + ")")
                    elif c.solution.get_status() in timstat:
                        timlimhits += 1
                        expected = Times.starttim + TIMLIM
                        if expected > tim:
                            raise Exception("Wallclock time limit expired too early (" +
                                            str(expected) + " vs. " +
                                            str(tim) + ")")
                    else:
                        raise Exception("Unexpected status " +
                                        str(c.solution.get_status()))

    if detlimhits == 0:
        raise Exception("Never hit a deterministic time limit")
    if timlimhits == 0:
        raise Exception("Never hit a wallclock time limit")
    print("Test passed")
