#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: sched_calendar.py
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
(Optimization Programming Language) .mod file, with data provided directly
within the Python script. The data is defined as a dictionary, where the keys
represent variable names and the values contain the corresponding data. It also
illustrates how to extract OPL variables from the model.
"""

from ordered_set import OrderedSet
import pandas as pd
from docplex.cp.model import CpoModel


def main():

    mod_file_path = "./data/sched_calendar.mod"

    with open(mod_file_path, "r") as file:
        mod = file.read()

    print("mod file content:\n\n", mod)

    data = {
        "NbHouses": 5,
        "WorkerNames": OrderedSet(["Joe", "Jim"]),
        "TaskNames": OrderedSet([
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
        "Duration": [
            35, 15, 40, 15, 5,
            10, 5, 10, 5, 5
        ],
        "Worker": {
            "masonry": "Joe",
            "carpentry": "Joe",
            "plumbing": "Jim",
            "ceiling": "Jim",
            "roofing": "Joe",
            "painting": "Jim",
            "windows": "Jim",
            "facade": "Joe",
            "garden": "Joe",
            "moving": "Jim",
        },
        "Precedences": OrderedSet([
            ("masonry", "carpentry"),
            ("masonry", "plumbing"),
            ("masonry", "ceiling"),
            ("carpentry", "roofing"),
            ("ceiling", "painting"),
            ("roofing", "windows"),
            ("roofing", "facade"),
            ("plumbing", "facade"),
            ("roofing", "garden"),
            ("plumbing", "garden"),
            ("windows", "moving"),
            ("facade", "moving"),
            ("garden", "moving"),
            ("painting", "moving"),
        ]),
        "Breaks": {
            "Joe": OrderedSet([
                (5, 14), (19, 21), (26, 28), (33, 35), (40, 42),
                (47, 49), (54, 56), (61, 63), (68, 70), (75, 77),
                (82, 84), (89, 91), (96, 98), (103, 105), (110, 112),
                (117, 119), (124, 133), (138, 140), (145, 147),
                (152, 154), (159, 161), (166, 168), (173, 175),
                (180, 182), (187, 189), (194, 196), (201, 203),
                (208, 210), (215, 238), (243, 245), (250, 252),
                (257, 259), (264, 266), (271, 273), (278, 280),
                (285, 287), (292, 294), (299, 301), (306, 308),
                (313, 315), (320, 322), (327, 329), (334, 336),
                (341, 343), (348, 350), (355, 357), (362, 364),
                (369, 378), (383, 385), (390, 392), (397, 399),
                (404, 406), (411, 413), (418, 420), (425, 427),
                (432, 434), (439, 441), (446, 448), (453, 455),
                (460, 462), (467, 469), (474, 476), (481, 483),
                (488, 490), (495, 504), (509, 511), (516, 518),
                (523, 525), (530, 532), (537, 539), (544, 546),
                (551, 553), (558, 560), (565, 567), (572, 574),
                (579, 602), (607, 609), (614, 616), (621, 623),
                (628, 630), (635, 637), (642, 644), (649, 651),
                (656, 658), (663, 665), (670, 672), (677, 679),
                (684, 686), (691, 693), (698, 700), (705, 707),
                (712, 714), (719, 721), (726, 728),
            ]),
            "Jim": OrderedSet([
                (5, 7), (12, 14), (19, 21), (26, 42), (47, 49),
                (54, 56), (61, 63), (68, 70), (75, 77), (82, 84),
                (89, 91), (96, 98), (103, 105), (110, 112),
                (117, 119), (124, 126), (131, 133), (138, 140),
                (145, 147), (152, 154), (159, 161), (166, 168),
                (173, 175), (180, 182), (187, 189), (194, 196),
                (201, 225), (229, 231), (236, 238), (243, 245),
                (250, 252), (257, 259), (264, 266), (271, 273),
                (278, 280), (285, 287), (292, 294), (299, 301),
                (306, 315), (320, 322), (327, 329), (334, 336),
                (341, 343), (348, 350), (355, 357), (362, 364),
                (369, 371), (376, 378), (383, 385), (390, 392),
                (397, 413), (418, 420), (425, 427), (432, 434),
                (439, 441), (446, 448), (453, 455), (460, 462),
                (467, 469), (474, 476), (481, 483), (488, 490),
                (495, 497), (502, 504), (509, 511), (516, 518),
                (523, 525), (530, 532), (537, 539), (544, 546),
                (551, 553), (558, 560), (565, 581), (586, 588),
                (593, 595), (600, 602), (607, 609), (614, 616),
                (621, 623), (628, 630), (635, 637), (642, 644),
                (649, 651), (656, 658), (663, 665), (670, 672),
                (677, 679), (684, 686), (691, 693), (698, 700),
                (705, 707), (712, 714), (719, 721), (726, 728),
            ]),
        },
    }

    mdl = CpoModel()
    mdl.build_opl_model(mod, data)
    sol = mdl.solve()

    if sol:
        print("Objective Value with created data dictionary:", sol.get_objective_value())
    else:
        print("No solution found")

    # Extracting Model Variables
    all_var = mdl.get_opl_var()
    print("opl variables:", all_var.variable_names)

    # Extracting Variable Solutions
    all_var_sol = all_var.from_solution(sol)

    print("\n\nsolution - itvs:")
    for i in all_var_sol.itvs.keys():
        for j in all_var_sol.itvs[i]:
            print("itvs[", i, "][", j, "]:", all_var_sol.itvs[i][j])

    print("\n\nsolution - SequenceVar_245:",
          [v.get_name() for v in all_var_sol.SequenceVar_245])

    print("\n\nsolution - SequenceVar_242:",
          [v.get_name() for v in all_var_sol.SequenceVar_242])


if __name__ == "__main__":
    main()