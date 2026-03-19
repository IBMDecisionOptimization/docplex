#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: admipex9.py
 
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Inject heuristic solutions from the generic callback.

Optimizes an all binary MIP problem.

See admipex2.py for an implementation using legacy callbacks.

To run this example, the user must specify a problem file.

You can run this example at the command line by

    python admipex9.py <filename>
"""
from math import fabs
import sys
import traceback

import cplex
from cplex.callbacks import Context


class HeuristicCallback():
    """A generic callback class that injects heuristic solutions.

    Everything is setup in the invoke function that is called by CPLEX.
    """

    def __init__(self, obj):
        """Create a new callback instance.

        obj is a dense objective vector.
        """
        self.times_called = 0
        self._obj = obj

    def round_down(self, context):
        """Heuristic motivated by knapsack constrained problems.

        Rounding down all fractional values will give an integer solution
        that is feasible, since all constraints are <= with positive
        coefficients.
        """
        self.times_called += 1

        # Get solution to the relaxation.
        x = context.get_relaxation_point()
        cols = len(self._obj)
        objrel = context.get_relaxation_objective()

        for j in range(cols):
            if x[j] > 0.0:
                # Set the fractional variables to zero.
                frac = x[j] % 1
                frac = min(1 - frac, frac)
                if frac > 1.0e-6:
                    objrel -= x[j] * self._obj[j]
                    x[j] = 0.0

        # Post the rounded solution.
        context.post_heuristic_solution(
            [list(range(cols)), x],
            objrel,
            Context.solution_strategy.check_feasible)

    def invoke(self, context):
        """Implements the required invoke method.

        This is the method that we have to implement to fulfill the
        generic callback contract. CPLEX will call this method during the
        solution process at the places that we asked for.
        """
        assert context.in_relaxation(), \
            "Callback called in an unexpected context {}".format(
                context.get_id())
        try:
            self.round_down(context)
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


def admipex9(filename):
    """Use the generic callback for optimizing a MIP problem."""
    with cplex.Cplex(filename) as c:

        if c.variables.get_num() != c.variables.get_num_binary():
            print("Problem contains non-binary variables, exiting")
            return

        # Set up to use generic callback.
        contextmask = 0
        contextmask |= Context.id.relaxation

        heuristiccb = HeuristicCallback(c.objective.get_linear())
        c.set_callback(heuristiccb, contextmask)

        c.parameters.mip.tolerances.mipgap.set(1.0e-6)

        # Disable heuristics so that our callback has a chance to make a
        # difference.
        c.parameters.mip.strategy.heuristicfreq.set(-1)

        # Optimize the problem and obtain solution.
        c.solve()

        solution = c.solution

        # solution.get_status() returns an integer code.
        print("Solution status = ", solution.get_status(), ":", end=' ')
        # The following line prints the corresponding string.
        print(solution.status[solution.get_status()])
        print("Objective value = ", solution.get_objective_value())
        print()

        # Write out the solution.
        x = solution.get_values(0, c.variables.get_num() - 1)
        for j in range(c.variables.get_num()):
            if fabs(x[j]) > 1.0e-10:
                print("Column %d: Value = %17.10g" % (j, x[j]))

        print("round_down was called ", heuristiccb.times_called, "times")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: admipex9.py filename")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        sys.exit(-1)
    admipex9(sys.argv[1])
