# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------
"""
Given source-destination pairs with specified flow requirements, information must travel along a single path per pair, with at most I
intermediate nodes. This example of communications Network Design problem involves determining the optimal physical network to transmit 
information between a given set of nodes. The objective is to minimize the total cost, which includes a fixed cost for each active link 
and a linear cost based on capacity, while ensuring that capacity limits are not exceeded.

Problem Constraints:
    1. We are given a set of N nodes <0, 1, ..., N-1>. A set of source / destination pairs <i,j> indicate that we want to send information 
    from i to j. The amount of this information (the flow from i to j) is indicated by F_ij > 0. If <i,j> is not a source destination 
    pair, then F_ij = 0. Between any source i and destination j, the information may flow directly on an arc which goes from i to j or 
    pass through intermediate nodes. The path must go through a maximum of I intermediate nodes, which is a global parameter of the system.

    2. The network is not "dynamic" in the sense that some of the information can be sent along one path from i to j and the the rest 
    along another path.

    3. The capacity Cab of a bi-directional link between nodes a and b is a real number determined by the maximum flow in either 
    direction, calculated by summing the flows of all source-destination pairs that use paths passing through the link. Load of link a-b 
    should be less than the capacity of the same link.

Objective"
    The objective is to determine the lowest-cost network by minimizing the total cost of the links, where the cost of each link is 
    proportional to its capacity and includes a fixed cost if the link is active; both the fixed cost and the linear cost coefficient vary 
    based on the specific nodes, reflecting differences in link lengths within the real network.

To run this example from the command line, use

    python network.py 8 2
"""
import numpy as np
from numpy.random import default_rng
import scipy,sys
from docplex.mp.model import Model
import pandas as pd

# -----------------------------------------------------------------------
# Initialize the problem data
# -----------------------------------------------------------------------
def generate_problem(num_nodes):

    # Generates a network problem with nodes, flow between nodes, link capacities, fixed costs, and variable costs.
    # various parameter for generating flow values, capacity, fixed cost, and variable cost 
    SOURCE_DEST_FRACTION = 0.25
    num_sd_pairs = int(0.5 + SOURCE_DEST_FRACTION * N * (N - 1))
    node_mass_generator = scipy.stats.uniform(1, 10)
    fixed_cost_coef = 10.0
    rng = default_rng()
    mass = node_mass_generator.rvs(size=num_nodes, random_state=rng)
    sd = set()
    while len(sd) < num_sd_pairs:
        i = rng.integers(num_nodes)
        j = rng.integers(num_nodes-1)
        j += (i == j)
        sd.add((i,j))
    FLOW_WINDOW = np.array([0.5, 2.0])
    CAP_WINDOW = FLOW_WINDOW * 10
    flow = np.array([[ ((i,j) in sd) * mass[i] * mass[j] * rng.uniform(*FLOW_WINDOW) for j in range(num_nodes) ] for i in range(num_nodes) ])
    x = rng.uniform(0, 1, size = num_nodes)
    y = rng.uniform(0, 1, size = num_nodes)
    distance = lambda a,b : ((x[a] - x[b])**2 + (y[a] - y[b])**2)**0.5
    cap_max = np.array([[ mass[i] * mass[j] * rng.uniform(*CAP_WINDOW) / (1 + distance(i, j)) 
                         for j in range(num_nodes) ] for i in range(num_nodes)])
    fix = fixed_cost_coef * np.array([[ rng.uniform(*FLOW_WINDOW) * (0.1 + distance(i, j)) 
                                       for j in range(num_nodes) ] for i in range(num_nodes)])
    variable = np.array([[ rng.uniform(*FLOW_WINDOW) * (0.1 + distance(i, j)) for j in range(num_nodes) ] for i in range(num_nodes)])
    print("No of Nodes = ",N)
    I, J, Fij, Fji, Cij, Fixij, Varij = ([] for _ in range(7))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i < j:
                # Note: cap_max[j,i], fix[j,i] and variable[j,i] are never used (j > i)
                I.append(i), J.append(j), Fij.append(flow[i, j]), Fji.append(flow[j, i]), Cij.append(cap_max[i, j]), 
                Fixij.append(fix[i, j]), Varij.append(variable[i, j])
    df = pd.DataFrame({'i':I, 'j':J, 'flow(i->j)':Fij, 'flow(j->i)':Fji, 'capacity_max(i<->j)':Cij, 
                   'fixed_cost_for_linking(i<->j)':Fixij, 'var_cost_per_value_of_capacity(i<->j)':Varij})
    print('Synthetic data for flow, capacity, fixed costs, and variable costs:')
    print(df)
    Flowij, max_capacity, fixed_cost, var_cost = {}, {}, {}, {}
    for i, j, fij, fji, cij, fixij, varij in zip(I, J, Fij, Fji, Cij, Fixij, Varij):
        Flowij.update({(i, j): fij, (j, i): fji})
        max_capacity.update({(i, j): cij, (j, i): cij})
        fixed_cost.update({(i, j): fixij, (j, i): fixij})
        var_cost.update({(i, j): varij, (j, i): varij}) 
    return Flowij, max_capacity, fixed_cost, var_cost

def network(N,Flowij, max_capacity, fixed_cost, var_cost):
    #-----------------------------------------------------------------------------
    # Creating the model and defining decision variables.
    #-----------------------------------------------------------------------------
    mdl = Model(name='network_flow')

    # Binary decision variable dictionary to determine which links (a-b) are used for a given source-destination pair (i-j).
    flow_ij_ab = mdl.binary_var_dict([(i, j, a, b) for i in nodes for j in nodes for a in nodes for b in nodes if a != b and i!=j ],
                                    name='flow_ij_ab')

    # Continuous decision variable dictionary to determine the load in link a-b.
    load_ab = mdl.continuous_var_dict([(a, b) for a in nodes for b in nodes if a != b],name='load_ab', lb=1e-4)


    # Decision variable to determine which links should have a physical connection.
    link_ab = mdl.binary_var_dict([(a, b) for a in nodes for b in nodes if a != b],name='link_ab')

    #-----------------------------------------------------------------------------
    # Setting the objective function for the model
    #-----------------------------------------------------------------------------
    objective_fun=[]
    for i in nodes:
        for j in nodes:
            if i!=j:
                term = link_ab[i, j]*fixed_cost[i, j] +  var_cost[i, j] * load_ab[i,j]
                objective_fun.append(term)            
    mdl.minimize(mdl.sum(objective_fun))

    #-----------------------------------------------------------------------------
    # Setting the constrains for the model
    #-----------------------------------------------------------------------------
    # Constraint to limit the maximum number of intermediate nodes and ensure a single path for each source-destination pair.
    for i in nodes:
        for j in nodes:
            if i!=j:
                if Flowij[(i,j)]>0.0:
                    mdl.add(mdl.sum(flow_ij_ab[(i,j,a,b)] for a in nodes for b in nodes if a!=b) <= Max_inter+1)
                    mdl.add(mdl.sum(flow_ij_ab[(i,j,i,b)] for b in nodes if b!=i) ==1)
                    mdl.add(mdl.sum(flow_ij_ab[(i,j,a,j)] for a in nodes if a!=j) ==1)
                    for k in nodes:
                        inflow = mdl.sum(flow_ij_ab[i, j, a, k] for a in nodes if a != k)
                        outflow = mdl.sum(flow_ij_ab[i, j, k, b] for b in nodes if k != b)
                        if k!=i and k!=j:
                            mdl.add(inflow - outflow == 0)
                mdl.add(flow_ij_ab[(i,j,a,b)]<= link_ab[(a,b)] for a in nodes for b in nodes if a!=b)
                mdl.add(load_ab[(i,j)] == mdl.max(
                    mdl.sum(flow_ij_ab[(a,b,i,j)]*Flowij[(a,b)] for a in nodes for b in nodes if a!=b and Flowij[(a,b)]),
                    mdl.sum(flow_ij_ab[(a,b,j,i)]*Flowij[(a,b)] for a in nodes for b in nodes if a!=b and Flowij[(a,b)])

                ))
                mdl.add(load_ab[(i,j)] <= max_capacity[(i,j)] * link_ab[i,j])    # Maximum capacity constraint
    return mdl, flow_ij_ab, load_ab

#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------
def display_results(mdl,flow_ij_ab,load_ab):

    print("\nSolving model....")
    solution = mdl.solve(log_output=True)
    solution.objective_value
    # solution.display()
    solution.solve_details
    print(mdl.solve_details)
    if solution:
        print('\nSolution Found: \n')
        for i in nodes:
            for j in nodes:
                if i!=j and Flowij[(i,j)]:
                        path_order = []
                        path = [(a,b) for a in nodes for b in nodes if a!=b if flow_ij_ab[(i,j,a,b)].solution_value]
                        print('i->j: ',i,'->',j,', Flow: ', Flowij[(i,j)])
                        for arc in path:
                            if i==arc[0]:
                                path_order.append(arc)    #finding the first arc in the path i->j 
                        for _ in range(len(path)-1):
                            for arc in path:
                                if path_order[-1][1] == arc[0]:
                                    path_order.append(arc)   # ordering the path
                        print('Path: ', path_order)
                        print('Load: ',[load_ab[tuple(p)].solution_value for p in path_order])
                        print('Max Capacity: ',[max_capacity[tuple(p)] for p in path_order])
                        print('\n')
    else:
        print("Search terminated by limit, no solution found.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Specify the number nodes and Maximum intermediate nodes")
        print("Usage: python network.py <No of nodes> <Maximum Intermediate Nodes>")
        print("Eg: python network 8 2")
        sys.exit(-1)
    else:
        N = int(sys.argv[1])
        nodes = range(N)
        Max_inter = int(sys.argv[2])
    Flowij, max_capacity, fixed_cost, var_cost = generate_problem(N)
    mdl,flow_ij_ab,load_ab = network(N,Flowij, max_capacity, fixed_cost, var_cost)
    display_results(mdl,flow_ij_ab,load_ab)