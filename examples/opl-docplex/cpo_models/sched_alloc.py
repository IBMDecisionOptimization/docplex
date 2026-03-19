#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: sched_alloc.py
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
This example script demonstrates how to build and solve an OPL (Optimization
Programming Language) model where the model is defined inline as a string, and
the data is provided directly within the same Python script. The data is
supplied as a combination of a dictionary and individual variables passed
as keyword arguments. The example also shows how to retrieve OPL variables
from the solved model.
"""

from ordered_set import OrderedSet
import pandas as pd
from docplex.cp.model import CpoModel


def main():

    mod = r"""
// --------------------------------------------------------------------------
// Licensed Materials - Property of IBM
//
// 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55
// Copyright IBM Corporation 1998, 2013. All Rights Reserved.
//
// Note to U.S. Government Users Restricted Rights:
// Use, duplication or disclosure restricted by GSA ADP Schedule
// Contract with IBM Corp.
// --------------------------------------------------------------------------

using CP;

{string} Tasks = ...;
int durations[Tasks] = ...;
int start[Tasks] = ...;

{string} Groups = ...;
int maxUnusedWorkers[Groups] = ...;

{string} mayperform[Tasks] = ...;

tuple OptTask {
    string task;
    string group;
}

{OptTask} optTasks = { <t,g> | t in Tasks, g in mayperform[t] };

{string} Workers = ...;
{string} workers[Groups] = ...;

dvar interval tasks[t in Tasks] size durations[t];
dvar interval opttasks[optTasks] optional;
dvar interval worker[Workers];

cumulFunction group[g in Groups] =
    sum (w in workers[g]) pulse(worker[w], 1)
  - sum (<t,g> in optTasks) pulse(opttasks[<t,g>], 1);

minimize max(w in Workers) lengthOf(worker[w]);

subject to {
    forall(t in Tasks)
        startOf(tasks[t]) == start[t];

    forall(t in Tasks)
        alternative(tasks[t], all(<t,g> in optTasks) opttasks[<t,g>]);

    forall(g in Groups) {
        0 <= group[g];
        group[g] <= maxUnusedWorkers[g];
    }
};
"""

    data = {
        "Tasks": OrderedSet([
            "masonry",
            "carpentry",
            "plumbing",
            "ceiling",
            "roofing",
            "painting",
            "windows",
            "facade",
            "garden",
            "moving",
        ]),

        "durations": [7, 3, 8, 3, 1, 2, 1, 2, 1, 1],

        "start": [0, 7, 7, 7, 10, 10, 11, 15, 15, 17],

        "Groups": OrderedSet(["g1", "g2", "g3"]),

        "maxUnusedWorkers": [2, 1, 1],

        "mayperform": {
            "masonry": OrderedSet(["g1", "g2"]),
            "carpentry": OrderedSet(["g1", "g3"]),
            "plumbing": OrderedSet(["g2"]),
            "ceiling": OrderedSet(["g1", "g3"]),
            "roofing": OrderedSet(["g1", "g3"]),
            "painting": OrderedSet(["g2", "g3"]),
            "windows": OrderedSet(["g1", "g3"]),
            "facade": OrderedSet(["g1", "g2"]),
            "garden": OrderedSet(["g1", "g2", "g3"]),
            "moving": OrderedSet(["g1", "g3"]),
        },
    }

    Workers = OrderedSet(["Thomas", "Brett", "Matthew", "Scott", "Bill"])

    workers = {
        "g1": OrderedSet(["Thomas", "Brett", "Matthew"]),
        "g2": OrderedSet(["Scott"]),
        "g3": OrderedSet(["Bill"]),
    }

    mdl = CpoModel()
    mdl.build_opl_model(mod, data, Workers=Workers, workers=workers)

    sol = mdl.solve()

    if sol:
        print("Objective Value with created data dictionary:",
              sol.get_objective_value())
    else:
        print("No solution found")

    # Extracting Model Variables
    all_Var = mdl.get_opl_var()
    print("opl variables:", all_Var.variable_names)

    # Extracting Variable Solutions
    all_Var_sol = all_Var.from_solution(sol)

    print("\n solution: worker - ")
    for i in all_Var_sol.worker.keys():
        print("worker[", i, "]:", all_Var_sol.worker[i])

    print("\n solution: opttasks - ")
    for i in all_Var_sol.opttasks.keys():
        print("opttasks[", i, "]:", all_Var_sol.opttasks[i])

    print("\n solution: tasks - ")
    for i in all_Var_sol.tasks.keys():
        print("tasks[", i, "]:", all_Var_sol.tasks[i])


if __name__ == "__main__":
    main()