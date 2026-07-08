#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: quadassign.py
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
This example script explains how to build and solve a model using an OPL
(Optimization Programming Language) .mod file, with data loaded from an Excel
file (from multiple sheets). The data is defined as a dictionary, where the keys
represent variable names and the values contain the corresponding data. It also
illustrates how to extract OPL variables from the model.
"""

import pandas as pd
from docplex.cp.model import CpoModel


def main():

    mod_file_path = "data/quadassign.mod"

    with open(mod_file_path, "r") as file:
        mod = file.read()

    print("mod file content:\n\n", mod)

    distances_xls = pd.read_excel(
        "./data/quadassign.xls",
        header=None,
        sheet_name="distances"
    )
    print("data - distances (from quadassign.xls):", distances_xls)

    flow_xls = pd.read_excel(
        "./data/quadassign.xls",
        header=None,
        sheet_name="flow"
    )
    print("data - flow (from quadassign.xls):", flow_xls)

    data = {
        "nbPerm": 19,
        "dist": distances_xls.values.tolist(),
        "flow": flow_xls.values.tolist(),
    }

    mdl = CpoModel()
    mdl.build_opl_model(mod, data)

    sol = mdl.solve(TimeLimit=30)

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

    print("solution:")
    for i in all_Var_sol.perm.keys():
        print("perm[", i, "]:", all_Var_sol.perm[i])


if __name__ == "__main__":
    main()