#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: xsteel.py
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
(Optimization Programming Language) .mod file, with data loaded from an Excel
file. The data is defined as a dictionary, where the keys represent variable
names and the values contain the corresponding data. The solution is then
written to another Excel file. It also illustrates how to extract OPL variables
from the model.
"""

from ordered_set import OrderedSet
import pandas as pd
from docplex.mp.model_reader import ModelReader


def main():

    mod_file_path = "./data/xsteel.mod"

    with open(mod_file_path, "r") as file:
        mod = file.read()

    print("mod file content:\n\n", mod)

    xsteel_xls = pd.read_excel("./data/xsteel.xls", header=None)
    print("Reading data from excel:", xsteel_xls)

    Products = xsteel_xls[0][1:3]
    TimePeriods = xsteel_xls[0][16:20]
    Rate = xsteel_xls[1][1:3]
    Inv0 = xsteel_xls[2][1:3]
    Avail = xsteel_xls[1][16:20]
    Market = xsteel_xls.loc[10:11, 1:4]
    Prodcost = xsteel_xls[3][1:3]
    Invcost = xsteel_xls[4][1:3]
    Revenue = xsteel_xls.loc[6:7, 1:4]

    data = {
        "Products": OrderedSet(Products.values.tolist()),
        "TimePeriods": OrderedSet(TimePeriods.values.tolist()),
        "Rate": [float(x) for x in Rate.values.tolist()],
        "Inv0": [float(x) for x in Inv0.values.tolist()],
        "Avail": [float(x) for x in Avail.values.tolist()],
        "Market": [[float(x) for x in row] for row in Market.values.tolist()],
        "Prodcost": [float(x) for x in Prodcost.values.tolist()],
        "Invcost": [float(x) for x in Invcost.values.tolist()],
        "Revenue": [[float(x) for x in row] for row in Revenue.values.tolist()],
    }

    mdl = ModelReader.build_opl_model(mod, data)
    sol = mdl.solve()

    if sol:
        print(
            "Objective Value with created data dictionary:",
            sol.get_objective_value(),
        )
    else:
        print("No solution found")

    all_Var = mdl.get_opl_var()
    print("opl variable names:", all_Var.variable_names)

    all_Var_sol = all_Var.from_solution(sol)

    print("Solution of variable Sell:", all_Var_sol.Sell)
    print("Solution of variable Inv:", all_Var_sol.Inv)
    print("Solution of variable Make:", all_Var_sol.Make)

    Sell = [
        [
            all_Var_sol.Sell["bands"][x]
            for x in all_Var.Sell[list(all_Var.Sell.keys())[0]].keys()
        ],
        [
            all_Var_sol.Sell["coils"][x]
            for x in all_Var.Sell[list(all_Var.Sell.keys())[1]].keys()
        ],
    ]

    Inv = [
        [
            all_Var_sol.Inv["bands"][x]
            for x in all_Var.Inv[list(all_Var.Inv.keys())[0]].keys()
        ],
        [
            all_Var_sol.Inv["coils"][x]
            for x in all_Var.Inv[list(all_Var.Inv.keys())[1]].keys()
        ],
    ]

    Make = [
        [
            all_Var_sol.Make["bands"][x]
            for x in all_Var.Make[list(all_Var.Make.keys())[0]].keys()
        ],
        [
            all_Var_sol.Make["coils"][x]
            for x in all_Var.Make[list(all_Var.Make.keys())[1]].keys()
        ],
    ]

    df_sell = pd.DataFrame(Sell)
    df_inv = pd.DataFrame(Inv)
    df_make = pd.DataFrame(Make)

    # Write to Excel with multiple sheets
    with pd.ExcelWriter("./data/output.xlsx", engine="xlsxwriter") as writer:
        df_sell.to_excel(writer, index=False, sheet_name="Sell")
        df_inv.to_excel(writer, index=False, sheet_name="Inv")
        df_make.to_excel(writer, index=False, sheet_name="Make")

    print("Excel file 'output.xlsx' created successfully.")


if __name__ == "__main__":
    main()