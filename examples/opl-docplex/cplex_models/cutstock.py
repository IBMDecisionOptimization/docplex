#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: cutstock.py
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
(Optimization Programming Language) mod content as string, with data given as **kwargs.
ie, Individual data variables passed as keyword arguments. This examples shows an
usage of IndexedSet & FrozenList. It also illustrates how to extract OPL 
variables from the model.
"""
from docplex.mp.model_reader import ModelReader
from docplex.util.collections import IndexedSet, FrozenList

mod = '''
int RollWidth = ...;
int NbItems = ...;

range Items = 1..NbItems;
int Size[Items] = ...;
int Amount[Items] = ...;

// used in column generation
float Duals[Items] = ...;


tuple  pattern {
   key int id;
   int cost;
   int fill[Items];
}


{pattern} Patterns = ...;

dvar float Cut[Patterns] in 0..1000000;


minimize
  sum( p in Patterns ) 
    p.cost * Cut[p];

subject to {
  forall( i in Items ) 
    ctFill: 
      sum( p in Patterns )
         p.fill[i] * Cut[p] >= Amount[i];
}
'''

NbItems = 5
RollWidth = 110
Size = [20, 45, 50, 55, 75]
Amount = [48, 35, 24, 10, 8]

Patterns = IndexedSet([ (0, 1, FrozenList([1, 0, 0, 0, 0])),
(1, 1, FrozenList([0, 1, 0, 0, 0])),
(2, 1, FrozenList([0, 0, 1, 0, 0])),
(3, 1, FrozenList([0, 0, 0, 1, 0])),
(4, 1, FrozenList([0, 0, 0, 0, 1])) ])
Duals = [0.0, 0.0, 0.0, 0.0, 0.0]


mdl = ModelReader.build_opl_model(mod,NbItems = NbItems,RollWidth = RollWidth, Size=Size, Amount=Amount, Patterns=Patterns, Duals=Duals)

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

for i in all_Var_sol.Cut:
    print('Cut [',i,']: ',all_Var_sol.Cut[i])