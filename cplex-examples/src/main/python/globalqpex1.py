#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: globalqpex1.py
 
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
Reading and optimizing a QP or MIQP problem.

Demonstrates options for convex and nonconvex problems by setting
parameters.

The user has to choose the method on the command line:

   python globalqpex1.py <filename> c     optimum for convex problem
   python globalqpex1.py <filename> f     first order solution (only for
                                          continuous problems)
   python globalqpex1.py <filename> g     global optimum
"""
import sys

import cplex
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes


def globalqpex1(filename, method):
    c = cplex.Cplex(filename)

    if (c.get_problem_type() != c.problem_type.QP and
            c.get_problem_type() != c.problem_type.MIQP):
        print("Input file is not a QP or MIQP.  Exiting.")
        return

    target = c.parameters.optimalitytarget.values

    if method == "c":
        c.parameters.optimalitytarget.set(target.optimal_convex)
    elif method == "f":
        c.parameters.optimalitytarget.set(target.first_order)
    elif method == "g":
        c.parameters.optimalitytarget.set(target.optimal_global)
    else:
        raise ValueError('method must be one of "c", "f", or "g"')

    try:
        c.solve()
    except CplexSolverError as e:
        if (method == "c" and e.args[2] == error_codes.CPXERR_Q_NOT_POS_DEF):
            if c.get_problem_type() == c.problem_type.MIQP:
                print("Problem is not convex. Use argument g to get "
                      "global optimum.")
            else:
                print("Problem is not convex. Use argument f to get "
                      "local optimum or g to get global optimum")
        elif (method == "f" and
              e.args[2] == error_codes.CPXERR_NOT_FOR_MIP and
              c.get_problem_type() == c.problem_type.MIQP):
            print("Problem is a MIP, cannot compute local optima "
                  "satifying the first order KKT.")
            print("Use argument g to get the global optimum.")
        else:
            print("Exception raised during solve")
        return

    # solution.get_status() returns an integer code
    status = c.solution.get_status()
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

    print("Solution status = ", status, ":", end=' ')
    # the following line prints the status as a string
    print(c.solution.status[status])
    print("Solution method = ", s_method, ":", end=' ')
    print(c.solution.method[s_method])

    if s_type == c.solution.type.none:
        print("No solution available")
        return
    print("Objective value = ", c.solution.get_objective_value())

    x = c.solution.get_values(0, c.variables.get_num() - 1)
    # because we're querying the entire solution vector,
    # x = c.solution.get_values()
    # would have the same effect
    for j in range(c.variables.get_num()):
        print("Column %d: Value = %17.10g" % (j, x[j]))

    infeas = c.solution.get_float_quality(
        c.solution.quality_metric.max_primal_infeasibility)
    print("Maximum bound violation = ", infeas)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ["c", "f", "g"]:
        print("Usage: globalqpex1.py filename algorithm")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        print("  algorithm  one of the letters")
        print("             c optimum for convex problem")
        print("             f first order solution (only continuous problems)")
        print("             g global optimum")
        sys.exit(-1)
    globalqpex1(sys.argv[1], sys.argv[2])
