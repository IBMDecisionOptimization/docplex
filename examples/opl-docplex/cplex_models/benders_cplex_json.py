#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: benders_cplex_json.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------

"""
This example script demonstrates how to build and solve a model using an OPL
(Optimization Programming Language) mod content as string, with data loaded from a json file.
The json data is loaded as a dictionary, where the keys represent variable names
and the values contain the corresponding data. It also illustrates how to extract
OPL variables from the model.
"""

from docplex.mp.model_reader import ModelReader
import json

mod = '''
int d1 = ...;
int d2 = ...;

range R1 = 1..d1;
range R2 = 1..d2;

range dim  = 1..d1*d2;
int Costs[i in R2][j in R1] = ...;

dvar float X[R2][R1];

dvar boolean Y[R1];

int bendersPartition[i in R2][j in R1] = i;
int bendersPartition2[i in R2] = i;

minimize sum(i in R2, j in R1) Costs[i][j]*X[i][j] + sum(i in R1) Y[i];
subject to{
forall(i in R2)
  sum(j in R1) X[i][j] ==1;
forall(i in R2, j in R1)
  X[i][j] - Y[j] <= 0;
}
'''
# building model from imported json
with open('./data/data_benders.json', "r") as f:
    d = json.load(f)

mdl = ModelReader.build_opl_model(mod,data = d)
sol = mdl.solve()

if sol:
    print(
        "Objective Value with created data dictionary:",
        sol.get_objective_value(),
    )
else:
    print("No solution found")

# Extracting OPL variables
all_Var = mdl.get_opl_var()
print("printing all variables:", all_Var.variable_names)
all_Var_sol = all_Var.from_solution(sol)
for i in all_Var_sol.X:
    print('X [',i,']: ',all_Var_sol.X[i])
    print()