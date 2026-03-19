#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: foodcsv.py
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
(Optimization Programming Language) .mod file, with data loaded from a csv file.
The data is defined as a dictionary, where the keys represent variable names
and the values contain the corresponding data. It also illustrates how to extract
OPL variables from the model.
"""

from docplex.mp.model_reader import ModelReader
from ordered_set import OrderedSet
import pandas as pd


def main():

    mod_file_path = "./data/foodcsv.mod"

    with open(mod_file_path, "r") as file:
        mod = file.read()

    print("mod file content:\n\n", mod)

    products_df = ["v1", "v2", "o1", "o2", "o3"]

    cost_df = pd.read_csv("./data/cost.csv")

    data = {
        "NbMonths": 6,
        "Products": OrderedSet(products_df),
        "Cost": {
            i + 1: cost_df.loc[i, products_df].tolist()
            for i in range(len(cost_df))
        },
    }

    # 'Cost' is structured as a dictionary where:
    #   - Keys represent months (1-based indexing),
    #   - Values are lists of product costs for that month.
    # This format aligns with the .mod file declaration:
    #   float Cost[Months][Products] = ...;

    mdl = ModelReader.build_opl_model(mod, data)
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

    all_Var.x102  # accessing variable x102

    # Extracting the solutions
    all_Var_sol = all_Var.from_solution(sol)

    print("solution of variable x202:", all_Var_sol.x202)

    print("solution of variable Produce:")
    for i in all_Var_sol.Produce:
        print(all_Var_sol.Produce[i])


if __name__ == "__main__":
    main()