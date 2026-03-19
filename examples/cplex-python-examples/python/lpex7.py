#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: lpex7.py
 
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
Reading and optimizing a problem.  Printing names with the
answer.  This is a modification of lpex2.py.

The user has to to choose the method on the command line:

   python lpex7.py <filename> o     cplex default
   python lpex7.py <filename> p     primal simplex
   python lpex7.py <filename> d     dual simplex
   python lpex7.py <filename> b     barrier
   python lpex7.py <filename> h     barrier with crossover
   python lpex7.py <filename> s     sifting
   python lpex7.py <filename> c     concurrent
"""
import sys

import cplex
from cplex.exceptions import CplexSolverError


def lpex7(filename, method):
    c = cplex.Cplex(filename)

    alg = c.parameters.lpmethod.values

    if method == "o":
        c.parameters.lpmethod.set(alg.auto)
    elif method == "p":
        c.parameters.lpmethod.set(alg.primal)
    elif method == "d":
        c.parameters.lpmethod.set(alg.dual)
    elif method == "b":
        c.parameters.lpmethod.set(alg.barrier)
        c.parameters.solutiontype.set(
            c.parameters.solutiontype.values.non_basic);
    elif method == "h":
        c.parameters.lpmethod.set(alg.barrier)
    elif method == "s":
        c.parameters.lpmethod.set(alg.sifting)
    elif method == "c":
        c.parameters.lpmethod.set(alg.concurrent)
    else:
        raise ValueError(
            'method must be one of "o", "p", "d", "b", "h", "s" or "c"')

    try:
        c.solve()
    except CplexSolverError:
        print("Exception raised during solve")
        return

    # solution.get_status() returns an integer code
    status = c.solution.get_status()
    print("Solution status = ", status, ":", end=' ')
    # the following line prints the status as a string
    print(c.solution.status[status])
    if status == c.solution.status.unbounded:
        print("Model is unbounded")
        return
    if status == c.solution.status.infeasible:
        print("Model is infeasible")
        return
    if status == c.solution.status.infeasible_or_unbounded:
        print("Model is infeasible or unbounded")
        return

    s_method = c.solution.get_method()
    s_type = c.solution.get_solution_type()

    print("Solution method = ", s_method, ":", end=' ')
    print(c.solution.method[s_method])

    if s_type == c.solution.type.none:
        print("No solution available")
        return
    print("Objective value = ", c.solution.get_objective_value())

    if s_type == c.solution.type.basic:
        basis = c.solution.basis.get_basis()[0]
    else:
        basis = None

    print()
    x = c.solution.get_values(0, c.variables.get_num() - 1)

    for j in range(c.variables.get_num()):
        print(c.variables.get_names(j) + ": %17.10g\t" % (x[j]))
        if basis is not None:
            if basis[j] == c.solution.basis.status.at_lower_bound:
                print("  Nonbasic at lower bound")
            elif basis[j] == c.solution.basis.status.basic:
                print("  Basic")
            elif basis[j] == c.solution.basis.status.at_upper_bound:
                print("  Nonbasic at upper bound")
            elif basis[j] == c.solution.basis.status.free_nonbasic:
                print("  Superbasic, or free variable at zero")
            else:
                print("   Bad basis status")

    infeas = c.solution.get_float_quality(
        c.solution.quality_metric.max_primal_infeasibility)
    print("Maximum bound violation = ", infeas)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ["o", "p", "d", "b",
                                                 "h", "s", "c"]:
        print("Usage: lpex7.py filename algorithm")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        print("  algorithm  one of the letters")
        print("             o default")
        print("             p primal simplex")
        print("             d dual simplex")
        print("             b barrier")
        print("             h barrier with crossover")
        print("             s sifting")
        print("             c concurrent")
        sys.exit(-1)
    lpex7(sys.argv[1], sys.argv[2])
