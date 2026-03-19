#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: bendersatsp2.py
 
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
r"""
Example bendersatsp2.py solves a flow MILP model for an Asymmetric
Traveling Salesman Problem (ATSP) instance through Benders decomposition.

The arc costs of an ATSP instance are read from an input file. The flow
MILP model is decomposed into a master ILP and a worker LP.

The master ILP is then solved by adding Benders' cuts during the
branch-and-cut process via the generic callback interface.

The example allows the user to decide if Benders' cuts have to be
separated:

a) Only to separate integer infeasible solutions.
In this case, Benders' cuts are treated as lazy constraints.

b) Also to separate fractional infeasible solutions.
In this case, Benders' cuts are also treated as user cuts.


To run this example from the command line, use

   python bendersatsp2.py {0|1} [filename]
where
    0         Indicates that Benders' cuts are only used as lazy
              constraints, to separate integer infeasible solutions.
    1         Indicates that Benders' cuts are also used as user cuts,
              to separate fractional infeasible solutions.

    filename  Is the name of the file containing the ATSP instance (arc
              costs).  If filename is not specified, the instance
              ../../../examples/data/atsp.dat is read


ATSP instance defined on a directed graph G = (V, A)
- V = {0, ..., n-1}, V0 = V \ {0}
- A = {(i,j) : i in V, j in V, i != j }
- forall i in V: delta+(i) = {(i,j) in A : j in V}
- forall i in V: delta-(i) = {(j,i) in A : j in V}
- c(i,j) = traveling cost associated with (i,j) in A

Flow MILP model

Modeling variables:
forall (i,j) in A:
   x(i,j) = 1, if arc (i,j) is selected
          = 0, otherwise
forall k in V0, forall (i,j) in A:
   y(k,i,j) = flow of the commodity k through arc (i,j)

Objective:
minimize sum((i,j) in A) c(i,j) * x(i,j)

Degree constraints:
forall i in V: sum((i,j) in delta+(i)) x(i,j) = 1
forall i in V: sum((j,i) in delta-(i)) x(j,i) = 1

Binary constraints on arc variables:
forall (i,j) in A: x(i,j) in {0, 1}

Flow constraints:
forall k in V0, forall i in V:
   sum((i,j) in delta+(i)) y(k,i,j) -
       sum((j,i) in delta-(i)) y(k,j,i) = q(k,i)
   where q(k,i) =  1, if i = 0
                = -1, if k == i
                =  0, otherwise

Capacity constraints:
forall k in V0, for all (i,j) in A: y(k,i,j) <= x(i,j)

Nonnegativity of flow variables:
forall k in V0, for all (i,j) in A: y(k,i,j) >= 0
"""
from math import fabs
import sys
import traceback

import cplex
from inputdata import read_dat_file


class ProbData():
    """Data class to read an ATSP instance from an input file"""

    def __init__(self, filename):

        # read the data in filename
        self.arc_cost = read_dat_file(filename)[0]
        self.num_nodes = len(self.arc_cost)

        # check data consistency
        for i in range(self.num_nodes):
            if len(self.arc_cost[i]) != self.num_nodes:
                print("ERROR: Data file '%s' contains inconsistent data\n" %
                      filename)
                raise Exception("data file error")
            self.arc_cost[i][i] = 0.0


class WorkerLP():
    """This class builds the worker LP (i.e., the dual of flow
    constraints and capacity constraints of the flow MILP) and allows to
    separate violated Benders' cuts.
    """

    # The constructor sets up the Cplex instance to solve the worker LP,
    # and creates the worker LP (i.e., the dual of flow constraints and
    # capacity constraints of the flow MILP)
    #
    # Modeling variables:
    # forall k in V0, i in V:
    #    u(k,i) = dual variable associated with flow constraint (k,i)
    #
    # forall k in V0, forall (i,j) in A:
    #    v(k,i,j) = dual variable associated with capacity constraint (k,i,j)
    #
    # Objective:
    # minimize sum(k in V0) sum((i,j) in A) x(i,j) * v(k,i,j)
    #          - sum(k in V0) u(k,0) + sum(k in V0) u(k,k)
    #
    # Constraints:
    # forall k in V0, forall (i,j) in A: u(k,i) - u(k,j) <= v(k,i,j)
    #
    # Nonnegativity on variables v(k,i,j)
    # forall k in V0, forall (i,j) in A: v(k,i,j) >= 0
    #
    def __init__(self, num_nodes):

        # Set up Cplex instance to solve the worker LP
        cpx = cplex.Cplex()
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)

        # Turn off the presolve reductions and set the CPLEX optimizer
        # to solve the worker LP with primal simplex method.
        cpx.parameters.preprocessing.reduce.set(0)
        cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.primal)

        cpx.objective.set_sense(cpx.objective.sense.minimize)

        # Create variables v(k,i,j) forall k in V0, (i,j) in A
        # For simplicity, also dummy variables v(k,i,i) are created.
        # Those variables are fixed to 0 and do not contribute to
        # the constraints.
        v = []
        for k in range(1, num_nodes):
            v.append([])
            for i in range(num_nodes):
                v[k - 1].append([])
                for j in range(num_nodes):
                    var_name = "v." + str(k) + "." + str(i) + "." + str(j)
                    v[k - 1][i].append(cpx.variables.get_num())
                    cpx.variables.add(obj=[0.0],
                                      lb=[0.0],
                                      ub=[cplex.infinity],
                                      names=[var_name])
                cpx.variables.set_upper_bounds(v[k - 1][i][i], 0.0)

        # Create variables u(k,i) forall k in V0, i in V
        u = []
        for k in range(1, num_nodes):
            u.append([])
            for i in range(num_nodes):
                var_name = "u." + str(k) + "." + str(i)
                u[k - 1].append(cpx.variables.get_num())
                obj = 0.0
                if i == 0:
                    obj = -1.0
                if i == k:
                    obj = 1.0
                cpx.variables.add(obj=[obj],
                                  lb=[-cplex.infinity],
                                  ub=[cplex.infinity],
                                  names=[var_name])

        # Add constraints:
        # forall k in V0, forall (i,j) in A: u(k,i) - u(k,j) <= v(k,i,j)
        for k in range(1, num_nodes):
            for i in range(num_nodes):
                for j in range(0, num_nodes):
                    if i != j:
                        thevars = []
                        thecoefs = []
                        thevars.append(v[k - 1][i][j])
                        thecoefs.append(-1.0)
                        thevars.append(u[k - 1][i])
                        thecoefs.append(1.0)
                        thevars.append(u[k - 1][j])
                        thecoefs.append(-1.0)
                        cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
                            senses=["L"],
                            rhs=[0.0])

        self.cpx = cpx
        self.v = v
        self.u = u
        self.num_nodes = num_nodes
        self.cut_lhs = None
        self.cut_rhs = None

    def separate(self, x_sol, x):
        """This method separates Benders' cuts violated by the current x
        solution.

        Violated cuts are found by solving the worker LP.
        """
        cpx = self.cpx
        v = self.v
        num_nodes = self.num_nodes
        violated_cut_found = False

        # Update the objective function in the worker LP:
        # minimize sum(k in V0) sum((i,j) in A) x(i,j) * v(k,i,j)
        #          - sum(k in V0) u(k,0) + sum(k in V0) u(k,k)
        thevars = []
        thecoefs = []
        for k in range(1, num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    thevars.append(v[k - 1][i][j])
                    thecoefs.append(x_sol[i][j])
        cpx.objective.set_linear(zip(thevars, thecoefs))

        # Solve the worker LP
        cpx.solve()

        # A violated cut is available iff the solution status is unbounded
        if cpx.solution.get_status() == cpx.solution.status.unbounded:

            # Get the violated cut as an unbounded ray of the worker LP
            ray = cpx.solution.advanced.get_ray()

            # Compute the cut from the unbounded ray. The cut is:
            # sum((i,j) in A) (sum(k in V0) v(k,i,j)) * x(i,j) >=
            # sum(k in V0) u(k,0) - u(k,k)
            num_arcs = num_nodes * num_nodes
            cut_vars_list = []
            cut_coefs_list = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    thecoef = 0.0
                    for k in range(1, num_nodes):
                        v_k_i_j_index = (k - 1) * num_arcs + i * num_nodes + j
                        if ray[v_k_i_j_index] > 1e-03:
                            thecoef = thecoef + ray[v_k_i_j_index]
                    if thecoef > 1e-03:
                        cut_vars_list.append(x[i][j])
                        cut_coefs_list.append(thecoef)
            cut_lhs = cplex.SparsePair(ind=cut_vars_list, val=cut_coefs_list)

            v_num_vars = (num_nodes - 1) * num_arcs
            cut_rhs = 0.0
            for k in range(1, num_nodes):
                u_k_0_index = v_num_vars + (k - 1) * num_nodes
                if fabs(ray[u_k_0_index]) > 1e-03:
                    cut_rhs = cut_rhs + ray[u_k_0_index]
                u_k_k_index = v_num_vars + (k - 1) * num_nodes + k
                if fabs(ray[u_k_k_index]) > 1e-03:
                    cut_rhs = cut_rhs - ray[u_k_k_index]

            self.cut_lhs = cut_lhs
            self.cut_rhs = cut_rhs
            violated_cut_found = True

        return violated_cut_found


class ATSPCallback():
    """Callback function for the ATSP problem.

    This callback can do two different things:
       - Separate Benders cuts at fractional solutions as user cuts
       - Separate Benders cuts at integer solutions as lazy constraints

    Everything is setup in the invoke function that
    is called by CPLEX.
    """

    def __init__(self, num_threads, num_nodes, x):
        self.num_threads = num_threads
        self.cutlhs = None
        self.cutrhs = None
        self.num_nodes = num_nodes
        self.x = x
        # Create workerLP for Benders' cuts separation
        self.workers = [None] * num_threads

    def separate_user_cuts(self, context, worker):
        """Separate Benders cuts at fractional solutions as user cuts."""
        # Get the current x solution
        sol = []
        for i in range(self.num_nodes):
            sol.append([])
            sol[i] = context.get_relaxation_point(self.x[i])

        # Benders' cut separation
        if worker.separate(sol, self.x):
            cutlhs = worker.cut_lhs
            cutrhs = worker.cut_rhs
            cutmanagement = cplex.callbacks.UserCutCallback.use_cut.purge
            context.add_user_cut(cut=cutlhs, sense='G', rhs=cutrhs,
                                 cutmanagement=cutmanagement, local=False)

    def separate_lazy_constraints(self, context, worker):
        """Separate Benders cuts at integer solutions as lazy constraints."""
        # We only work with bounded models
        if not context.is_candidate_point():
            raise Exception('Unbounded solution')
        # Get the current x solution
        sol = []
        for i in range(self.num_nodes):
            sol.append([])
            sol[i] = context.get_candidate_point(self.x[i])

        # Benders' cut separation
        if worker.separate(sol, self.x):
            cutlhs = worker.cut_lhs
            cutrhs = worker.cut_rhs
            context.reject_candidate(
                constraints=[cutlhs, ], senses='G', rhs=[cutrhs, ])

    def invoke(self, context):
        """Whenever CPLEX needs to invoke the callback it calls this
        method with exactly one argument: an instance of
        cplex.callbacks.Context.
        """
        try:
            thread_id = context.get_int_info(
                cplex.callbacks.Context.info.thread_id)
            if context.get_id() == cplex.callbacks.Context.id.thread_up:
                self.workers[thread_id] = WorkerLP(self.num_nodes)
            elif context.get_id() == cplex.callbacks.Context.id.thread_down:
                self.workers[thread_id] = None
            elif context.get_id() == cplex.callbacks.Context.id.relaxation:
                self.separate_user_cuts(context, self.workers[thread_id])
            elif context.get_id() == cplex.callbacks.Context.id.candidate:
                self.separate_lazy_constraints(
                    context, self.workers[thread_id])
            else:
                print("Callback called in an unexpected context {}".format(
                    context.get_id()))
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


def create_master_ilp(cpx, x, data):
    """This function creates the master ILP (arc variables x and degree
    constraints).

    Modeling variables:
    forall (i,j) in A:
       x(i,j) = 1, if arc (i,j) is selected
              = 0, otherwise

    Objective:
    minimize sum((i,j) in A) c(i,j) * x(i,j)

    Degree constraints:
    forall i in V: sum((i,j) in delta+(i)) x(i,j) = 1
    forall i in V: sum((j,i) in delta-(i)) x(j,i) = 1

    Binary constraints on arc variables:
    forall (i,j) in A: x(i,j) in {0, 1}
    """

    arc_cost = data.arc_cost
    num_nodes = data.num_nodes

    cpx.objective.set_sense(cpx.objective.sense.minimize)

    # Create variables x(i,j) for (i,j) in A
    # For simplicity, also dummy variables x(i,i) are created.
    # Those variables are fixed to 0 and do not partecipate to
    # the constraints.
    for i in range(num_nodes):
        x.append([])
        for j in range(num_nodes):
            var_name = "x." + str(i) + "." + str(j)
            x[i].append(cpx.variables.get_num())
            cpx.variables.add(obj=[arc_cost[i][j]],
                              lb=[0.0], ub=[1.0], types=["B"],
                              names=[var_name])
        cpx.variables.set_upper_bounds(x[i][i], 0)

    # Add the out degree constraints.
    # forall i in V: sum((i,j) in delta+(i)) x(i,j) = 1
    for i in range(num_nodes):
        thevars = []
        thecoefs = []
        for j in range(0, i):
            thevars.append(x[i][j])
            thecoefs.append(1)
        for j in range(i + 1, num_nodes):
            thevars.append(x[i][j])
            thecoefs.append(1)
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["E"], rhs=[1.0])

    # Add the in degree constraints.
    # forall i in V: sum((j,i) in delta-(i)) x(j,i) = 1
    for i in range(num_nodes):
        thevars = []
        thecoefs = []
        for j in range(0, i):
            thevars.append(x[j][i])
            thecoefs.append(1)
        for j in range(i + 1, num_nodes):
            thevars.append(x[j][i])
            thecoefs.append(1)
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(thevars, thecoefs)],
            senses=["E"], rhs=[1.0])


def bendersatsp(sep_frac_sols, filename):
    """Solves a flow MILP model for an Asymmetric Traveling Salesman
    Problem (ATSP) instance through Benders decomposition.
    """
    print("Benders' cuts separated to cut off: ", end=' ')
    if sep_frac_sols == "1":
        print("Integer and fractional infeasible solutions.")
        sep_frac_sols = True
    elif sep_frac_sols == "0":
        print("Only integer infeasible solutions.")
        sep_frac_sols = False
    else:
        raise ValueError('sep_frac_sols must be one of "0" or "1"')

    # Read arc costs from data file (9 city problem)
    data = ProbData(filename)

    # Create master ILP
    cpx = cplex.Cplex()
    x = []
    create_master_ilp(cpx, x, data)
    num_nodes = data.num_nodes

    num_threads = cpx.get_num_cores()
    cpx.parameters.threads.set(num_threads)

    atspcb = ATSPCallback(num_threads, num_nodes, x)
    contextmask = cplex.callbacks.Context.id.thread_up
    contextmask |= cplex.callbacks.Context.id.thread_down
    contextmask |= cplex.callbacks.Context.id.candidate
    if sep_frac_sols:
        contextmask |= cplex.callbacks.Context.id.relaxation
    cpx.set_callback(atspcb, contextmask)

    # Solve the model
    cpx.solve()

    solution = cpx.solution
    print()
    print("Solution status: ", solution.get_status())
    print("Objective value: ", solution.get_objective_value())

    if solution.get_status() == solution.status.MIP_optimal:
        # Write out the optimal tour
        succ = [-1] * num_nodes
        for i in range(num_nodes):
            sol = solution.get_values(x[i])
            for j in range(num_nodes):
                if sol[j] > 1e-03:
                    succ[i] = j
        print("Optimal tour:")
        i = 0
        while succ[i] != 0:
            print("%d, " % i, end=' ')
            i = succ[i]
        print(i)
    else:
        print("Solution status is not optimal")


def usage():
    """Prints usage statement."""
    print("""\
Usage:     bendersatsp2.py {0|1} [filename]
 0:        Benders' cuts only used as lazy constraints,
           to separate integer infeasible solutions.
 1:        Benders' cuts also used as user cuts,
           to separate fractional infeasible solutions.
 filename: ATSP instance file name.
           File ../../../examples/data/atsp.dat used if
           no name is provided.
""")


def main():
    """Set default arguments and parse command line."""
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        usage()
        sys.exit(-1)
    if sys.argv[1] not in ["0", "1"]:
        usage()
        sys.exit(-1)
    if len(sys.argv) == 3:
        filename = sys.argv[2]
    else:
        filename = "../../../examples/data/atsp.dat"
    bendersatsp(sys.argv[1][0], filename)


if __name__ == "__main__":
    main()
