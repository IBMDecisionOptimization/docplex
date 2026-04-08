#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: sched_flowshop.py
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
(Optimization Programming Language) model as string, with data given as json as
String. It also illustrates how to extract OPL variables from the model.
"""
from docplex.cp.model import CpoModel
import json

mod = '''
using CP;

int nbJobs = ...;
int nbMchs = ...;

range Jobs = 0..nbJobs-1;
range Mchs = 0..nbMchs-1; 

int OpDurations[j in Jobs][m in Mchs] = ...;

dvar interval itvs[j in Jobs][m in Mchs] size OpDurations[j][m];
dvar sequence mchs[m in Mchs] in all(j in Jobs) itvs[j][m];

execute {
  		cp.param.FailLimit = 10000;
}

minimize max(j in Jobs) endOf(itvs[j][nbMchs-1]);
subject to {
  forall (m in Mchs)
    noOverlap(mchs[m]);
  forall (j in Jobs, o in 0..nbMchs-2)
    endBeforeStart(itvs[j][o], itvs[j][o+1]);
}

execute {
  for (var j = 0; j <= nbJobs-1; j++) {
    for (var o = 0; o <= nbMchs-1; o++) {
      write(itvs[j][o].start + " ");
    }
    writeln("");
  }
}

'''

json_data = '''
{
  "nbJobs": 20,
  "nbMchs": 5,
  "OpDurations": [
    [62, 22, 77, 88, 39],
    [68, 94, 66, 57, 48],
    [57, 6, 20, 24, 86],
    [54, 37, 87, 50, 78],
    [50, 80, 62, 60, 58],
    [59, 78, 79, 93, 88],
    [36, 55, 10, 13, 43],
    [46, 81, 36, 13, 36],
    [83, 67, 39, 1, 88],
    [23, 54, 25, 8, 2],
    [2, 57, 82, 63, 16],
    [38, 20, 93, 15, 13],
    [82, 51, 66, 89, 63],
    [9, 34, 42, 42, 46],
    [76, 25, 13, 13, 23],
    [99, 34, 77, 24, 41],
    [76, 23, 96, 56, 84],
    [12, 94, 2, 5, 9],
    [13, 84, 57, 78, 72],
    [19, 86, 6, 58, 27]
  ]
}
'''
# Pass JSON data as string
mdl = CpoModel()
mdl.build_opl_model(mod,json_data)
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
for i in all_Var_sol.itvs:
    print('itvs [',i,']: ',all_Var_sol.itvs[i])