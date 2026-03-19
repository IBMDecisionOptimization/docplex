# --------------------------------------------------------------------------
# File: genericbranch.py
 
# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2019, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""Demonstrates how to perform customized branching using the generic
callback.

For any model with integer variables passed on the command line, the
code will solve the model using a simple customized branching strategy.
The branching strategy implemented here is most-infeasible branching,
i.e., the code always picks the integer variable that is most fractional
and then branches on it. If the biggest fractionality of all integer
variables is small then the code refrains from custom branching and lets
CPLEX decide.

See usage message below.
"""
import math
import sys

import cplex


def usage():
    """Print a usage message and exit."""
    print("""\
Usage: {0} filename...
  filename   Name of a file, or multiple files, with .mps, .lp, or .sav
             extension, and a possible, additional .gz extension.\
""".format(sys.argv[0]))
    sys.exit(2)


class BranchCallback:
    """Generic callback that implements most infeasible branching."""

    def __init__(self, x):
        """Creates a new callback instance.

        x - the list of indices of all non-continuous variables.
        """
        self._x = x
        self.calls = 0  # How often was the callback invoked?
        self.branches = 0  # How many branches did the callback create?

    def invoke(self, context):
        """Implements the required invoke method.

        This is the method that we have to implement to fulfill the
        generic callback contract. CPLEX will call this method during the
        solution process at the places that we asked for.
        """
        self.calls += 1

        # For sake of illustration, prune every node that has a depth
        # larger than 1000.
        depth = context.get_long_info(cplex.callbacks.Context.info.node_depth)
        if depth > 1000:
            context.prune_current_node()
            return

        # Get the status of the current relaxation.
        # The get_relaxation_status() method not only fetches the status,
        # but also makes sure the node LP is solved before returning the
        # status (this may trigger a re-solve of the node LP).
        # That status can be used to identify numerical issues, etc.
        status = context.get_relaxation_status()

        # Only branch if the current node relaxation could be solved to
        # optimality.
        # If there was any sort of trouble then don't do anything and
        # thus let CPLEX decide how to cope with that.
        if status != context.solution_status.optimal and status != context.solution_status.optimal_infeasible:
            return

        # Get the objective value of the current relaxation. We use this
        # as estimate for the new children to create.
        obj = context.get_relaxation_objective()

        # The node LP was solved to optimality. Grab the current
        # relaxation and find the most fractional variable.
        max_var = -1
        max_frac = 0.0
        max_val = 0.0
        for j, v in zip(self._x, context.get_relaxation_point(self._x)):
            intval = round(v)
            frac = abs(intval - v)

            if frac > max_frac:
                max_frac = frac
                max_var = j
                max_val = v

        # If the maximum fractionality of all integer variables is small
        # then don't create a custom branch. Instead let CPLEX decide how
        # to branch.
        min_frac = 0.1
        if max_frac > min_frac:
            # There is a variable with a sufficiently fractional value.
            # Branch on that variable.
            up = math.ceil(max_val)
            down = math.floor(max_val)
            branch_var = max_var

            # Create UP branch (branch_var >= up).
            up_child = context.make_branch(obj, [(branch_var, 'L', up)])
            self.branches += 1

            # Create DOWN branch (branch_var <= down).
            down_child = context.make_branch(obj, [(branch_var, 'U', down)])
            self.branches += 1


def genericbranch(model):
    """Branching with the generic callback for one model."""
    with cplex.Cplex(model) as cpx:
        ctype = cpx.variables.get_types()
        # Create a callback and pass as argument the indices of all
        # non-continuous variables.
        cb = BranchCallback([i for i, c in enumerate(ctype) if c != 'C'])

        # Register the callback with CPLEX and ask CPLEX to invoke it
        # only in the branching context.
        cpx.set_callback(cb, cplex.callbacks.Context.id.branching)

        # Limit the number of nodes.
        # The branching strategy implemented here is not smart so solving
        # even a simple MIP may turn out to take a long time.
        cpx.parameters.mip.limits.nodes.set(1000)

        # Solve the model and report some statistics.
        cpx.solve()
        print('Model {0} solved, status = {1}'
              .format(model, cpx.solution.get_status()))
        print('Callback was invoked {0} times and created {1} branches'
              .format(cb.calls, cb.branches))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
    for arg in sys.argv[1:]:
        genericbranch(arg)
