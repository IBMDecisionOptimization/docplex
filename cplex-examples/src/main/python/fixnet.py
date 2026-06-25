#!/usr/bin/python
# --------------------------------------------------------------------------
# File: fixnet.c
 
# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2018, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------

r"""
fixnet.py - Use indicator constraints to avoid numerical trouble in a
            fixed charge network flow problem.


Find a minimum cost flow in a fixed charge network flow problem.
The network is as follows:

       1 -- 3 ---> demand = 1,000,000
      / \  /
     /   \/
    0    /\
     \  /  \
      \/    \
       2 -- 4 ---> demand = 1

A fixed charge of one is incurred for every edge with non-zero flow,
with the exception of edge <1,4>, which has a fixed charge of ten.
The cost per unit of flow on an edge is zero, with the exception of
edge <2,4>, where the cost per unit of flow is five.
"""
import cplex


def fixnet():
    # Arc data
    orig = [0, 0, 1, 1, 2, 2]
    dest = [1, 2, 3, 4, 3, 4]
    unitcost = [0, 0, 0, 0, 0, 5]
    fixedcost = [1, 1, 1, 10, 1, 1]
    # Node data (negative demand is injected into network, positive demand
    #            is removed)
    demand = [-1000001, 0, 0, 1000000, 1]

    n = len(demand)  # Number of nodes
    m = len(orig)   # Number of arcs

    with cplex.Cplex() as cpx:
        # Create the flow constraints, one for each node.
        # Initially, all these constraints have an empty left-hand side.
        # The left-hand side will be populated when creating the flow variables.
        cpx.linear_constraints.add(lin_expr=None,
                                   senses=['G'] * n,
                                   rhs=demand,
                                   names=['flow%i' for i in range(n)])

        # Create the flow variables.
        # These are continuous variables in [0,inf[, hence we can use the
        # default values for arguments 'lb', 'ub', 'types'.
        # Along with creating the variables we also populate the left-hand
        # sides of the flow constraints. Variable x[i] corresponds to arc i,
        # which runs from orig[i] to dest[i]. Hence variable x[i] has
        # coefficient +1 in the constraints for orig[i] and coefficient -1
        # in the constraint for dest[i].
        x = list()
        for i in range(m):
            x += list(cpx.variables.add(obj=[unitcost[i]],
                                        names=['x%d%d' % (orig[i], dest[i])],
                                        columns=[cplex.SparsePair(ind=[orig[i], dest[i]],
                                                                  val=[-1.0,     1.0])]))

        # Create the fixed-charge variables.
        # These are binary variables, one for each arc.
        f = list(cpx.variables.add(obj=fixedcost,
                                   lb=[0] * m,
                                   ub=[1] * m,
                                   types=['B'] * m,
                                   names=['f%d%d' % (orig[i], dest[i]) for i in range(m)]))

        # Create the indicator constraints that state
        #  if f[i]=0 then x[i]<=0
        # Since the constraints x[i]<=0 shall only be active if f[i]=0, we
        # have to pass complemented=1 for each constraint.
        cpx.indicator_constraints.add_batch(lin_expr=[cplex.SparsePair(ind=[x[i]],
                                                                       val=[1.0]) for i in range(m)],
                                            sense=['L'] * m,
                                            rhs=[0.0] * m,
                                            indvar=f,
                                            complemented=[1] * m,
                                            name=['indicator%d' % i for i in range(m)])
        cpx.write('fixnet.lp')
        cpx.solve()
        print('Solution status:', cpx.solution.get_status())
        print('Solution value  = %f' % cpx.solution.get_objective_value())
        print('Solution vector:')
        for n, v in zip(cpx.variables.get_names(), cpx.solution.get_values()):
            print('%8s: %15.6f' % (n, v))


if __name__ == "__main__":
    fixnet()
