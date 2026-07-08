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
Dantzig-Wolfe decomposition for a multi-commodity transportation problem
based on AMPL sample multi1a.run, multi1.mod, and multi1.dat.

See also Mat-lab version multi.m.
"""
import cplex

# Populate data here
ORIG = ['GARY', 'CLEV', 'PITT']
DEST = ['FRA', 'DET', 'LAN', 'WIN', 'STL', 'FRE', 'LAF']
PROD = ['bands', 'coils', 'plate']

supply_list = \
[400,  700,  800,
800, 1600, 1800,
200,  300,  300]
i = 0
supply = {}
for p in PROD:
    for o in ORIG:
        supply[(o, p)] = supply_list[i]
        i = i + 1

demand_list = \
[300, 300, 100,  75, 650, 225, 250,
500, 750, 400, 250, 950, 850, 500,
100, 100,   0,  50, 200, 100, 250]
i = 0
demand = {}
for p in PROD:
    for d in DEST:
        demand[(d, p)] = demand_list[i]
        i = i + 1

limit = 625

cost_list = \
[30, 10,  8, 10, 11, 71,  6,
22,  7, 10,  7, 21, 82, 13,
19, 11, 12, 10, 25, 83, 15,
39, 14, 11, 14, 16, 82,  8,
27,  9, 12,  9, 26, 95, 17,
24, 14, 17, 13, 28, 99, 20,
41, 15, 12, 16, 17, 86,  8,
29,  9, 13,  9, 28, 99, 18,
26, 14, 17, 13, 31,104, 20]

i = 0
cost={}
for p in PROD:
    for o in ORIG:
        for d in DEST:
            cost[(o, d, p)] = cost_list[i]
            i = i + 1

phase = 1
nPROP = 0
price_convex = 1

# Define the initial subproblem here
subprob = cplex.Cplex()
subprob.parameters.read.datacheck.set(1)
price = {}
for o in ORIG:
    for d in DEST:
            price[(o, d)] = 0
subprob_obj =[]
subprob_colnames = []
for o in ORIG:
    for d in DEST:
         for p in PROD:
            subprob_obj.append(-price[(o, d)])
            subprob_colnames.append("Trans_" + o + "_" + d + "_" + p)
subprob.variables.add(obj = subprob_obj, names = subprob_colnames)
subprob.objective.set_name("Artif_Reduced_Cost")
supply_rmat = []
supply_rnames =[]
supply_rhs = []
for o in ORIG:
    for p in PROD:
        sameDest = []
        for d  in DEST:
            sameDest.append("Trans_" + o + "_" + d + "_" + p)
        ones = [1 for x in range(0,len(DEST))]
        supply_rmat.append([sameDest, ones])
        supply_rnames.append("supply_" + o + "_" + p)
        supply_rhs.append(supply[(o, p)])
subprob.linear_constraints.add(rhs = supply_rhs, senses = 'E'*len(supply_rnames), \
                                                names = supply_rnames, lin_expr = supply_rmat)

demand_rmat = []
demand_rnames =[]
demand_rhs = []
for d  in DEST:
    for p in PROD:
        sameOrig = []
        for o in ORIG:
            sameOrig.append("Trans_" + o + "_" + d + "_" + p)
        ones = [1 for x in range(0,len(ORIG))]
        demand_rmat.append([sameOrig, ones])
        demand_rnames.append("demand_" + d + "_" + p)
        demand_rhs.append(demand[(d, p)])
subprob.linear_constraints.add(rhs = demand_rhs, senses = 'E'*len(demand_rnames), \
                                                names = demand_rnames, lin_expr = demand_rmat)

# Define the initial master problem here
masterprob = cplex.Cplex()
masterprob.parameters.read.datacheck.set(1)
masterprob.variables.add(names = ["Excess"], obj = [1])
masterprob.objective.set_name("Artificial")
for i in range(1, 0):
    # Add weight's one by one
    masterprob.variables.add(names="Weight_"+str(i))

prop_ship = {}
for o in ORIG:
    for d in DEST:
        for n in range(1, nPROP+1):
            prop_ship[(o, d, n)] = 0
            for p in PROD:
                prop_ship[(o, d, n)] = prop_ship[(o, d, n)] + \
		     ubprob.Solution.get_objective("Trans_" + o + "_" + d + "_" + p)
prop_cost = ["0_is_never_used"]
for p in range(0,nPROP):
    prop_cost.append(0)
    for o in ORIG:
        for d in DEST:
            for p in PROD:
                prop_cost[p] = prop_cost[p] + cost[(o, d, p)] * \
		     subprob.Solution.get_objective("Trans_" + o + "_" + d + "_" + p)
multi_rmat = []
multi_rnames = []
for o in ORIG:
    for d in DEST:
        multi = []
        multi_colnames = []
        for n in range(0, nPROP):
            multi.append(prop_ship[(o, d, n)])
            multi_colnames.append("Weight_" + n)
        multi.append(-1)
        multi_colnames.append("Excess")
        multi_rmat.append([multi_colnames, multi])
        multi_rnames.append("Multi_" + o + "_" + d)
masterprob.linear_constraints.add(lin_expr  = multi_rmat, names = multi_rnames, \
		                 senses = "L" * len(multi_rnames), rhs = [limit] * len(multi_rnames))

masterprob.linear_constraints.add(lin_expr = [1] * nPROP, senses = "E", rhs = [1], names = ["Convex"])

# Main loop
while True:
    print("***PHASE " + str(phase) + " --- ITERATION " + str(nPROP+1) + "***")

    #subprob.write("subprob"+str(nPROP)+".lp")
    subprob.solve()

    print("subprob solution status = ", subprob.solution.get_status())
    print("subprob solution value  = ", subprob.solution.get_objective_value() - price_convex)
    Trans = {}
    for o in ORIG:
        for d  in DEST:
            for p in PROD:
                Trans[(o, d, p)] = subprob.solution.get_values("Trans_" + o + "_" + d + "_" + p)

    if phase == 1:
        if subprob.solution.get_objective_value() - price_convex >= -0.00001:
            print("\n*** NO FEASIBLE solution ***\n")
            break
    else:
        if subprob.solution.get_objective_value() - price_convex >= -0.00001:
            print("\n*** OPTIMAL solution ***\n")
            break

    nPROP = nPROP + 1
    for o in ORIG:
        for d in DEST:
            prop_ship[(o, d, nPROP)] = 0
            for p in PROD:
                prop_ship[(o, d, nPROP)] = prop_ship[(o, d, nPROP)] + Trans[(o, d, p)]

    prop_cost.append(0)
    # itertools.combinations is not available for python 2.6-; have to do this in old-fashion way
    for o in ORIG:
        for d in DEST:
            prop_cost[nPROP] = prop_cost[nPROP] + sum(cost[(o, d, p)] * Trans[(o, d, p)] \
										for p in PROD)

    # Update coefficients in master problem
    Weight_cind = []
    Weight_cval = []
    for o in ORIG:
        for d in DEST:
            for n in range(1, nPROP):
                masterprob.linear_constraints.set_coefficients("Multi_" + o + "_" + d,
								        n, prop_ship[(o, d, n)])
            # Need to add a column here for the nPROP column
            Weight_cind.append("Multi_" + o + "_" + d)
            Weight_cval.append(prop_ship[(o, d, nPROP)])
    Weight_cind.append("Convex")
    Weight_cval.append(1)
    masterprob.variables.add(names = ["Weight_" + str(nPROP)], columns = [[Weight_cind, Weight_cval]])
    if masterprob.objective.get_name() == "Total_Cost":
        masterprob.objective.set_linear("Weight_"+str(nPROP), prop_cost[nPROP])

    #masterprob.write("masterprob"+str(nPROP)+".lp")
    masterprob.solve()
    print("masterprob solution status = ", masterprob.solution.get_status())
    print("masterprob solution value  = ", masterprob.solution.get_objective_value())
    print("\n")
    if phase == 1:
        temp = masterprob.solution.get_values("Excess")
        if temp <= 0.00001:
            print("\nSETTING UP FOR PHASE 2\n\n")
            phase = 2
            # Update master problem
            masterprob.objective.set_name("Total_Cost")
            masterprob.objective.set_linear("Excess", 0)
            for n in range(1, nPROP+1):
                masterprob.objective.set_linear("Weight_"+str(n), prop_cost[n])
            masterprob.variables.set_lower_bounds("Excess", temp)
            masterprob.variables.set_upper_bounds("Excess", temp)
            # Update sub problem
            subprob.objective.set_name("Reduced_Cost")
            for o in ORIG:
                for d in DEST:
                    for p in PROD:
                        subprob.objective.set_linear("Trans_" + o + "_" + d + "_" + p, \
							     cost[(o, d, p)] - price[(o, d)])
            #masterprob.write("master.lp")
            masterprob.solve()
            print("masterprob solution status = ", masterprob.solution.get_status())
            print("masterprob solution value  = ", masterprob.solution.get_objective_value())
            print("\n")

    for o in ORIG:
        for d in DEST:
            price[(o, d)] = masterprob.solution.get_dual_values("Multi_" + o + "_" + d)
            for p in PROD:
                if subprob.objective.get_name() == "Artif_Reduced_Cost":
                    subprob.objective.set_linear("Trans_" + o + "_" + d + "_" + p,  \
						 - price[(o, d)])
                else:
                    subprob.objective.set_linear("Trans_" + o + "_" + d + "_" + p,  \
						 cost[(o, d, p)] - price[(o, d)])
    price_convex = masterprob.solution.get_dual_values("Convex")

print("***PHASE 3***\n\n")
opt_ship = {}
for o in ORIG:
    for d in DEST:
        opt_ship[(o, d)] = 0
        for n in range(1, nPROP+1):
            opt_ship[(o, d)] = opt_ship[(o, d)] + prop_ship[(o, d, n)] * \
						  masterprob.solution.get_values("Weight_"+str(n))
Master3 = cplex.Cplex()
for o in ORIG:
    for d in DEST:
            price[(o, d)] = 0
MasterIII_obj =[]
MasterIII_colnames = []
for o in ORIG:
    for d in DEST:
         for p in PROD:
            MasterIII_obj.append(cost[(o, d, p)])
            MasterIII_colnames.append("Trans_" + o + "_" + d + "_" + p)
Master3.variables.add(obj = MasterIII_obj, names = MasterIII_colnames)
Master3.objective.set_name("Opt_Cost")
supply_rmat = []
supply_rnames = []
supply_rhs = []
for o in ORIG:
    for p in PROD:
        sameDest = []
        for d  in DEST:
            sameDest.append("Trans_" + o + "_" + d + "_" + p)
        ones = [1 for x in range(0,len(DEST))]
        supply_rmat.append([sameDest, ones])
        supply_rnames.append("supply_" + o + "_" + p)
        supply_rhs.append(supply[(o, p)])
Master3.linear_constraints.add(rhs = supply_rhs, senses = 'E'*len(supply_rnames), \
                                                names = supply_rnames, lin_expr = supply_rmat)

demand_rmat = []
demand_rnames =[]
demand_rhs = []
for d  in DEST:
    for p in PROD:
        sameOrig = []
        for o in ORIG:
            sameOrig.append("Trans_" + o + "_" + d + "_" + p)
        ones = [1 for x in range(0,len(ORIG))]
        demand_rmat.append([sameOrig, ones])
        demand_rnames.append("demand_" + d + "_" + p)
        demand_rhs.append(demand[(d, p)])
Master3.linear_constraints.add(rhs = demand_rhs, senses = 'E'*len(demand_rnames), \
                                                names = demand_rnames, lin_expr = demand_rmat)

opt_multi_rmat = []
opt_multi_rnames = []
opt_multi_rhs = []
for o in ORIG:
    for d in DEST:
        opt_multi_list = []
        sameProd = []
        for p in PROD:
            sameProd.append("Trans_" + o + "_" + d + "_" + p)
        ones = [1 for x in range(0,len(PROD))]
        opt_multi_rmat.append([sameProd, ones])
        opt_multi_rnames.append("opt_multi_" + o + "_" + d)
        opt_multi_rhs.append(opt_ship[(o, d)])

Master3.linear_constraints.add(rhs = opt_multi_rhs, senses = 'E'*len(opt_multi_rnames), \
                                                names = opt_multi_rnames, lin_expr = opt_multi_rmat)
#Master3.write("Master3.lp")
Master3.solve()
objval = Master3.solution.get_objective_value()
print("Final solution status = ", Master3.solution.get_status())
print("Final objective value = ", objval)
assert abs(objval - 199500.0) < 1e-6, "was: {0}".format(objval)
