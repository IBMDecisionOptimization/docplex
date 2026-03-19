#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: multiobjex1.py
 
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
Reading and optimizing a multiobjective problem.

Demonstrates specifying parameter sets and accessing the results of the
various objectives.

Usage:

   python multiobjex1.py <filename> [<prmfile1> <prmfile2> ...]

The files prmfile1, ... are applied as paramsets for the different
objective priorities.
"""
import sys

from cplex import Cplex
from cplex.exceptions import CplexSolverError


def multiobjex1(filename, paramfiles=None):
    """Solve a multi-objective model."""
    with Cplex(filename) as c:
        if c.multiobj.get_num() == 1:
            print('Model is not multi-objective')
            return
        try:
            if paramfiles:
                # We need to read-in all the parameter sets
                paramsets = [c.create_parameter_set() for i in paramfiles]
                _ = [p.read(f) for p, f in zip(paramsets, paramfiles)]
                c.solve(paramsets=paramsets)
            else:
                c.solve()
        except CplexSolverError:
            print("Exception raised during solve")
            return

        # solution.get_status() returns an integer code
        status = c.solution.get_status()
        print(c.solution.status[status])
        if status == c.solution.status.multiobj_unbounded:
            print("Model is unbounded")
            return
        if status == c.solution.status.multiobj_infeasible:
            print("Model is infeasible")
            return
        if status == c.solution.status.multiobj_inforunbd:
            print("Model is infeasible or unbounded")
            return
        if status == c.solution.status.multiobj_stopped:
            print("Optimization was not finished")
            return


        print("Solution status = ", status, ":", end=' ')
        # the following line prints the status as a string
        print(c.solution.status[status])

        # Now print the values of the various objective for the
        # solution
        print('Objective values...')
        for i in range(c.multiobj.get_num()):
            print("Objective {} value = {}".
                  format(i, c.solution.multiobj.get_objective_value(i)))

        print()

        # Now print the objective values by priorities
        priorities = sorted(set([c.multiobj.get_priority(i)
                                 for i in range(c.multiobj.get_num())]),
                            reverse=True)
        print('Objective values by priorities...')
        objval_by_priority = c.solution.multiobj.get_objval_by_priority
        for p in priorities:
            print("Objective priority {} value = {}".
                  format(p, objval_by_priority(p)))



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(-1)
    elif len(sys.argv) == 2:
        multiobjex1(sys.argv[1])
    else:
        multiobjex1(sys.argv[1], sys.argv[2:])
