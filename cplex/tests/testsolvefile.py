# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Solve CPLEX models in various ways.

No command line arguments are required.
"""
import sys
import os
import time
import cplex
import cplex._internal._pycplex as CR
from cplex._internal._procedural import check_status


class NoOp(cplex.callbacks.BranchCallback):

    def __call__(self):
        pass
    

def by_row(c):
    nrows = c.linear_constraints.get_num()
    ncols = c.variables.get_num()
    ncoef = c.linear_constraints.get_num_nonzeros()

    sys.stderr.write("\nCHECK: problem statistics:\n\n")
    sys.stderr.write("".join(["CHECK: columns = :", str(ncols), "\n"]))
    sys.stderr.write("".join(["CHECK: rows    = :", str(nrows), "\n"]))
    sys.stderr.write("".join(["CHECK: nnz     = :", str(ncoef), "\n\n"]))

    start_time = time.time()
    sys.stderr.write("extracting problem data by rows ...\t")
    obj = c.objective.get_linear()
    vlb = c.variables.get_lower_bounds()
    vub = c.variables.get_upper_bounds()
    if c.get_problem_type() == c.problem_type.LP:
        vtp = ""
    else:
        vtp = c.variables.get_types()
    rhs = c.linear_constraints.get_rhs()
    sns = c.linear_constraints.get_senses()
    mat = c.linear_constraints.get_rows()
    rng = [0.0] * nrows
    for i, sense in enumerate(sns):
        if sense == "R":
            rng[i] = c.linear_constraints.get_range_values(i)
    obj_sen = c.objective.get_sense()
    sys.stderr.write("".join([str(time.time() - start_time), " seconds\n"]))

    start_time = time.time()
    sys.stderr.write("rebuilding problem by rows     ...\t")
    d = cplex.Cplex()
    if ncols != 0:
        d.variables.add(obj = obj, lb = vlb, ub = vub, types = vtp)
    d.objective.set_sense(obj_sen)
    d.linear_constraints.add(rhs = rhs, senses = sns, range_values = rng, lin_expr = mat)
    sys.stderr.write("".join([str(time.time() - start_time), " seconds\n"]))

def by_col(c):
    nrows = c.linear_constraints.get_num()
    ncols = c.variables.get_num()
    ncoef = c.linear_constraints.get_num_nonzeros()

    sys.stderr.write("\nCHECK: problem statistics:\n\n")
    sys.stderr.write("".join(["CHECK: columns = :", str(ncols), "\n"]))
    sys.stderr.write("".join(["CHECK: rows    = :", str(nrows), "\n"]))
    sys.stderr.write("".join(["CHECK: nnz     = :", str(ncoef), "\n\n"]))

    start_time = time.time()
    sys.stderr.write("extracting problem data by columns ...\t")
    rhs = c.linear_constraints.get_rhs()
    sns = c.linear_constraints.get_senses()
    vlb = c.variables.get_lower_bounds()
    vub = c.variables.get_upper_bounds()
    obj = c.objective.get_linear()
    if c.get_problem_type() == c.problem_type.LP:
        vtp = ""
    else:
        vtp = c.variables.get_types()
    mat = c.variables.get_cols()
    rng = [0.0] * nrows
    for i, sense in enumerate(sns):
        if sense == "R":
            rng[i] = c.linear_constraints.get_range_values(i)
    obj_sen = c.objective.get_sense()
    sys.stderr.write("".join([str(time.time() - start_time), " seconds\n"]))

    start_time = time.time()
    sys.stderr.write("rebuilding problem by columns     ...\t")
    d = cplex.Cplex()
    d.objective.set_sense(obj_sen)
    if nrows != 0:
        d.linear_constraints.add(rhs = rhs, senses = sns, range_values = rng)
    if ncols != 0:
        d.variables.add(obj = obj, lb = vlb, ub = vub, types = vtp, columns = mat)
    sys.stderr.write("".join([str(time.time() - start_time), " seconds\n"]))


def read_model(filename, rowwise, colwise, verbose):
    sys.stderr.write("".join(["Reading file '", filename, "' ...\t"]))
    start_time = time.time()
    c = cplex.Cplex(filename)
    if verbose:
        c.parameters.simplex.display.set(2)
    sys.stderr.write("".join([str(time.time() - start_time), " seconds\n"]))
    if rowwise:
        by_row(c)
    if colwise:
        by_col(c)
    return c

def is_optimal(c):
    s = c.solution.status
    status = c.solution.get_status()
    if status in [s.optimal, s.MIP_optimal, s.optimal_tolerance,
                  s.optimal_infeasible, s.optimal_populated,
                  s.optimal_populated_tolerance]:
        return True
    return status in [s.node_limit_feasible, s.solution_limit,
                      s.populate_solution_limit, s.fail_feasible,
                      s.mem_limit_feasible, s.fail_feasible_no_tree,
                      s.feasible] and c.solution.is_dual_feasible()

def is_infeasible(c):
    s = c.solution.status
    status = c.solution.get_status()
    return status in [s.infeasible, s.optimal_relaxed_sum,
                      s.optimal_relaxed_inf, s.optimal_relaxed_quad]

def is_mip(c):
    return c.get_problem_type() in [c.problem_type.MILP,
                                    c.problem_type.MIQP,
                                    c.problem_type.MIQCP,
                                    c.problem_type.fixed_MILP,
                                    c.problem_type.fixed_MIQP,]

def solvefile(filename, rowwise=False, doSolve=True, branch=False,
              colwise=False, verbose=False):
    c = read_model(filename, rowwise, colwise, verbose)

    if doSolve:
        start_time = time.time()
        sys.stderr.write("CHECK: solving problem ...\n")
        if branch:
            c.register_callback(NoOp)
        c.solve()
        if is_optimal(c):
            print("Optimal solution:  Objective = ", c.solution.get_objective_value())
            print("Solution time =    ", start_time - time.time(), " seconds", end=' ')
            print("   Iterations = ", c.solution.progress.get_num_iterations(), end=' ')
            if is_mip(c):
                print("   Nodes = ", c.solution.progress.get_num_nodes_processed())
            else:
                print()
        elif is_infeasible(c):
            print("Infeasible:")
            print("Solution time =    ", start_time - time.time(), " seconds", end=' ')
            print("   Iterations = ", c.solution.progress.get_num_iterations(), end=' ')
            if is_mip(c):
                print("   Nodes = ", c.solution.progress.get_num_nodes_processed())
            else:
                print()
        elif c.solution.get_status() == c.solution.status.abort_time_limit:
            print("Time limit exceeded:")
            print("Solution time =    ", start_time - time.time(), " seconds", end=' ')
            print("   Iterations = ", c.solution.progress.get_num_iterations(), end=' ')
            if is_mip(c):
                print("   Nodes = ", c.solution.progress.get_num_nodes_processed())
            else:
                print()
        else:
            sys.stderr.write("".join(["CHECK: solution status is ", c.solution.status[c.solution.get_status()], "\n"]))
            if c.solution.is_primal_feasible():
                sys.stderr.write("solution is primal feasible")
            else:
                sys.stderr.write("solution might not be primal feasible")
            if c.solution.is_dual_feasible():
                sys.stderr.write("solution is dual feasible")
            else:
                sys.stderr.write("solution might not be dual feasible")
        if is_infeasible(c) or c.solution.get_status() == c.solution.status.infeasible_or_unbounded:
            pass # calls to getIIS, some combination of conflict.refine and conflict.get calls
        else:
            sys.stderr.write("".join(["CHECK: solution value is ", str(c.solution.get_objective_value()), "\n"]))
            if is_mip(c):
                sys.stderr.write("".join(["CHECK: solution value is ", str(c.solution.MIP.get_best_objective()), "\n"]))

def usage():
    sys.stderr.write("".join(["CHECK: usage: " , sys.argv[0] , " [options] <filename>" , "\n"]))
    sys.stderr.write("".join(["CHECK:        -h        help message" , "\n"]))
    sys.stderr.write("".join(["CHECK:        -r        rowwwise build model" , "\n"]))
    sys.stderr.write("".join(["CHECK:        -c        columnwise build model" , "\n"]))
    sys.stderr.write("".join(["CHECK:        -b        call empty branch callback when solving MIP" , "\n"]))
    sys.stderr.write("".join(["CHECK:        -s        solve model" , "\n"]))
    sys.stderr.write("".join(["CHECK:        -v        verbose output" , "\n"]))
    sys.stderr.write("".join(["CHECK: Use default file ../../../examples/data/afiro.mps" , "\n"]))

def main():
    """The main function."""
    solvefile(filename="../../../examples/data/afiro.mps",
              rowwise=True,
              doSolve=True,
              branch=False,
              colwise=True,
              verbose=False)
    solvefile(filename="../../../examples/data/afiro.mps",
              rowwise=False,
              doSolve=True,
              branch=True,
              colwise=False,
              verbose=True)
    solvefile(filename="../../../examples/data/base.lp")
    solvefile(filename="../../../examples/data/case1.lp")
    solvefile(filename="../../../examples/data/case2.lp")
    solvefile(filename="../../../examples/data/case3.lp")
    solvefile(filename="../../../examples/data/case4.lp")
    solvefile(filename="../../../examples/data/case5.lp")
    solvefile(filename="../../../examples/data/qcp.lp")

    print()
    print("Maximum memory usage  --  ", end=' ')
    sys.stdout.flush()
    os.system("grep VmHWM /proc/%d/status" % (os.getpid(),))

if __name__ == '__main__':
    main()
