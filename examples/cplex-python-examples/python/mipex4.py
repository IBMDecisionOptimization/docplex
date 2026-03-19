#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: mipex4.py
 
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Reading in and optimizing a problem using a callback to log or interrupt
or an Aborter to interrupt.

To run this example, command line arguments are required:

    python mipex4.py filename option

 where

    filename  Name of the file, with .mps, .lp, or .sav
              extension, and a possible additional .gz
              extension.
    option is one of
       t to use the time-limit-gap callback
       l to use the logging callback
       a to use the aborter

Example:

    python mipex4.py myexample.mps l
"""
import sys
import time
from math import fabs

import cplex
from cplex.callbacks import MIPInfoCallback



class TimeLimitCallback(MIPInfoCallback):

    def __call__(self):
        if not self.aborted and self.has_incumbent():
            gap = 100.0 * self.get_MIP_relative_gap()
            timeused = self.get_time() - self.starttime
            if timeused > self.timelimit and gap < self.acceptablegap:
                print("Good enough solution at", timeused, "sec., gap =",
                      gap, "%, quitting.")
                self.aborted = True
                self.abort()


class LoggingCallback(MIPInfoCallback):

    def __call__(self):
        newincumbent = False
        hasincumbent = self.has_incumbent()
        if hasincumbent:
            incobjvalue = self.get_incumbent_objective_value()
            if fabs(self.lastincumbent - incobjvalue) > \
                    1e-5 * (1 + fabs(incobjvalue)):
                self.lastincumbent = incobjvalue
                newincumbent = True

        nodes = self.get_num_nodes()
        if nodes >= self.lastlog + 100 or newincumbent:
            if nodes >= self.lastlog + 100:
                self.lastlog = nodes

            walltime = self.get_time()
            dettime = self.get_dettime()

            if hasincumbent:
                incstr = "  Incumbent objective = " + \
                         str(self.get_incumbent_objective_value())
            else:
                incstr = ""
            print("Time = %.2f  Dettime = %.2f  Nodes = %d(%d)  "
                  "Best objective = %g%s"
                  % (walltime - self.timestart, dettime - self.dettimestart,
                     nodes, self.get_num_remaining_nodes(),
                     self.get_best_objective_value(), incstr))

        if newincumbent:
            incval = self.get_incumbent_values()
            print("New incumbent variable values:", incval)


def mipex4(filename, option):
    c = cplex.Cplex(filename)
    logging_cb = None

    if option == "t":
        timelim_cb = c.register_callback(TimeLimitCallback)
        timelim_cb.starttime = c.get_time()
        timelim_cb.timelimit = 1
        timelim_cb.acceptablegap = 10
        timelim_cb.aborted = False
    elif option == "l":
        # Set an overall node limit in case callback conditions
        # are not met.
        c.parameters.mip.limits.nodes.set(5000)
        logging_cb = c.register_callback(LoggingCallback)
        logging_cb.lastincumbent = 1e+75
        logging_cb.lastlog = -1e+75
        logging_cb.timestart = c.get_time()
        logging_cb.dettimestart = c.get_dettime()
        # Turn off CPLEX logging
        c.parameters.mip.display.set(0)
    elif option == "a":
        # Typically, you would pass the Aborter object to another thread
        # or pass it to an interrupt handler, and monitor for some event
        # to occur. When it does, call the Aborter's abort method.
        #
        # To illustrate its use without creating a thread or an interrupt
        # handler, abort immediately by calling abort before the solve.
        aborter = c.use_aborter(cplex.Aborter())
        aborter.abort()
    else:
        raise ValueError('option must be one of "t", "l", or "a"')

    c.solve()

    sol = c.solution

    print()
    # solution.get_status() returns an integer code
    print("Solution status = ", sol.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(sol.status[sol.get_status()])

    if sol.is_primal_feasible():
        print("Solution value  = ", sol.get_objective_value())
    else:
        print("No solution available.")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: mipex4.py filename option")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        print("  option is one of")
        print("     t to use the time-limit-gap callback")
        print("     l to use the logging callback")
        print("     a to use the aborter")
        sys.exit(-1)
    mipex4(sys.argv[1], sys.argv[2])
