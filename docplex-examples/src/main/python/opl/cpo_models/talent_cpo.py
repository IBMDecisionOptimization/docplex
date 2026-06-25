#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: talent_cpo.py
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
(Optimization Programming Language) model as string, with data given as
dictionary imported from a json file. It also illustrates how to extract OPL
variables from the model.
"""
from docplex.cp.model import CpoModel
import json


mod = '''
using CP;

execute{
	}
int numActors = ...;
range Actors = 1..numActors;
int actorPay[Actors] = ...;
int numScenes = ...;
range Scenes = 1..numScenes;
int sceneDuration[Scenes] = ...;

int actorInScene[Actors][Scenes]  = ...;

dvar int scene[Scenes] in Scenes;
dvar int slot[Scenes] in Scenes;


// First and last slots where each actor plays
dexpr int firstSlot[a in Actors] = min(s in Scenes:actorInScene[a][s] == 1) slot[s];
dexpr int lastSlot[a in Actors] = max(s in Scenes:actorInScene[a][s] == 1) slot[s];

// Expression for the waiting time for each actor
dexpr int actorWait[a in Actors] = sum(s in Scenes: actorInScene[a][s] == 0)  
   (sceneDuration[s] * (firstSlot[a] <= slot[s] && slot[s] <= lastSlot[a]));

// Expression representing the global cost
dexpr int idleCost = sum(a in Actors) actorPay[a] * actorWait[a];

minimize idleCost;
subject to {
   // use the slot-based secondary model
   inverse(scene, slot);
}

tuple slotSolutionT{ 
	int Scenes; 
	int value; 
};
{slotSolutionT} slotSolution = {<i0,slot[i0]> | i0 in Scenes};
tuple sceneSolutionT{ 
	int Scenes; 
	int value; 
};
{sceneSolutionT} sceneSolution = {<i0,scene[i0]> | i0 in Scenes};
'''
# building model from imported json
with open('data/data_talent.json', "r") as f:
    d = json.load(f)

mdl = CpoModel()
mdl.build_opl_model(mod,data = d)

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

for i in all_Var_sol.slot:
    print('slot [',i,']: ',all_Var_sol.slot[i])