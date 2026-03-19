#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: facility.py
 
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
Solve a capacitated facility location problem, potentially using Benders
decomposition.

The model solved here is

   minimize
       sum(j in locations) fixedCost[j]// open[j] +
       sum(j in locations) sum(i in clients) cost[i][j] * supply[i][j]
   subject to
       sum(j in locations) supply[i][j] == 1                    for each
                                                                client i
       sum(i in clients) supply[i][j] <= capacity[j] * open[j]  for each
                                                                location j
       supply[i][j] in [0,1]
       open[j] in {0, 1}

For further details see the usage() function.

You can run this example at the command line by

   python facility.py
"""
import sys

import cplex
from inputdata import read_dat_file


# Benders decomposition types used for solving the model.
NO_BENDERS = 1
AUTO_BENDERS = 2
ANNO_BENDERS = 3


def usage():
    """Prints a usage statement and exits the program."""
    print("""\
Usage: facility.py [options] [inputfile]
 where
   inputfile describes a capacitated facility location instance as in
   ../../../../examples/data/facility.dat. If no input file
   is specified read the file in example/data directory.
   Options are:
   -a solve problem with Benders letting CPLEX do the decomposition
   -b solve problem with Benders specifying a decomposition
   -d solve problem without using decomposition (default)
 Exiting...
""")
    sys.exit(2)


def facility(datafile, bendersopt):
    """Solve capacitated facility location problem."""
    # Read in data file. If no file name is given on the command line
    # we use a default file name. The data we read is
    # fixedcost  -- a list/array of facility fixed cost
    # cost       -- a matrix for the costs to serve each client by each
    #               facility
    # capacity   -- a list/array of facility capacity
    fixedcost, cost, capacity = read_dat_file(datafile)
    num_locations = len(fixedcost)
    num_clients = len(cost)

    # Create the modeler/solver.
    cpx = cplex.Cplex()

    # Create variables. We have variables
    # open_[j]        if location j is open.
    # supply[i][j]]   how much client i is supplied from location j
    open_ = list(cpx.variables.add(obj=fixedcost,
                                   lb=[0] * num_locations,
                                   ub=[1] * num_locations,
                                   types=["B"] * num_locations))
    supply = [None] * num_clients
    for i in range(num_clients):
        # Objective: Minimize the sum of fixed costs for using a location
        #            and the costs for serving a client from a specific
        #            location.
        supply[i] = list(cpx.variables.add(obj=cost[i],
                                           lb=[0.0] * num_locations,
                                           ub=[1.0] * num_locations))

    # Constraint: Each client i must be assigned to exactly one location:
    #   sum(j in nbLocations) supply[i][j] == 1  for each i in nbClients
    for i in range(num_clients):
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                ind=supply[i], val=[1.0] * num_locations)],
            senses=["E"],
            rhs=[1.0])

    # Constraint: For each location j, the capacity of the location must
    #             be respected:
    #   sum(i in nbClients) supply[i][j] <= capacity[j] * open_[j]
    for j in range(num_locations):
        ind = [supply[i][j] for i in range(num_clients)] + [open_[j]]
        val = [1.0] * num_clients + [-capacity[j]]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=ind, val=val)],
            senses=["L"],
            rhs=[0.0])

    # Setup Benders decomposition if required.
    if bendersopt == ANNO_BENDERS:
        # We specify the structure for doing a Benders decomposition by
        # telling CPLEX which variables are in the master problem using
        # annotations. By default variables are assigned value
        # CPX_BENDERS_MASTERVALUE+1 and thus go into the workers.
        # Variables open_[j] should go into the master and therefore
        # we assign them value CPX_BENDERS_MASTER_VALUE.
        mastervalue = cpx.long_annotations.benders_mastervalue
        idx = cpx.long_annotations.add(
            name=cpx.long_annotations.benders_annotation,
            defval=mastervalue + 1)
        objtype = cpx.long_annotations.object_type.variable
        cpx.long_annotations.set_values(idx, objtype,
                                        [(open_[x], mastervalue)
                                         for x in range(num_locations)])
        print("Solving with explicit Benders decomposition.")
    elif bendersopt == AUTO_BENDERS:
        # Let CPLEX automatically decompose the problem.  In the case of
        # a capacitated facility location problem the variables of the
        # master problem should be the integer variables.  By setting the
        # Benders strategy parameter to Full, CPLEX will put all integer
        # variables into the master, all continuous varibles into a
        # subproblem, and further decompose that subproblem, if possible.
        cpx.parameters.benders.strategy.set(
            cpx.parameters.benders.strategy.values.full)
        print("Solving with automatic Benders decomposition.")
    elif bendersopt == NO_BENDERS:
        print("Solving without Benders decomposition.")
    else:
        raise ValueError("invalid bendersopt argument")

    # Solve and display solution
    cpx.solve()
    print("Solution status =", cpx.solution.get_status_string())
    print("Optimal value:", cpx.solution.get_objective_value())
    tol = cpx.parameters.mip.tolerances.integrality.get()
    values = cpx.solution.get_values()
    for j in [x for x in range(num_locations) if values[open_[x]] >= 1.0 - tol]:
        print("Facility {0} is open, it serves clients {1}".format(
            j, " ".join([str(x) for x in range(num_clients)
                         if values[supply[x][j]] >= 1.0 - tol])))


def main():
    """Handles command line argument parsing."""
    filename = "../../../examples/data/facility.dat"
    benders = NO_BENDERS
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            if arg == "-a":
                benders = AUTO_BENDERS
            elif arg == "-b":
                benders = ANNO_BENDERS
            elif arg == "-d":
                benders = NO_BENDERS
            else:
                usage()
        else:
            filename = arg
    facility(datafile=filename, bendersopt=benders)


if __name__ == "__main__":
    main()
