#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: fleet.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# Copyright IBM Corporation 2026. All Rights Reserved.
# ---------------------------------------------------------------------------

"""
This example script explains how to build and solve a model using an OPL
(Optimization Programming Language) .mod file, with input data supplied
directly through Python code. It also illustrates how to extract OPL
variables from the model.

This example demonstrates a fleet assignment optimization problem using
CPLEX and DOcplex's OPL model integration.
"""

from docplex.mp.model_reader import ModelReader
from docplex.util.collections import IndexedSet
import pandas as pd
from collections import namedtuple


def main():

    # Read OPL model file
    mod_file_path = "data/fleet.mod"
    with open(mod_file_path, "r") as file:
        mod = file.read()

    print("mod file content:\n\n", mod)

    # Sets
    Airports = IndexedSet(
        [
            "Source", "ABZ", "AMS", "BGO", "BHX",
            "DUB", "EDI", "NWI", "SVG", "LBA",
            "EMA", "Sink",
        ]
    )

    Distance = IndexedSet(["Short", "Medium", "Long"])
    Fleets = IndexedSet(["D95", "M82", "M87"])

    MaxRefuel = 20
    Horizon = 60 * 24 * 7 - 1   # Minutes in a week
    MaxSpill = 0.1

    # Fleet Information
    FleetInfoElem = namedtuple(
        "FleetInfoElem",
        "aircrafts dist seats refuelT a b"
    )

    FleetInfo = {
        "D95": FleetInfoElem(8, "Short", 100, 12, 18000, 2),
        "M82": FleetInfoElem(5, "Medium", 150, 15, 25500, 3),
        "M87": FleetInfoElem(14, "Long", 200, 18, 35500, 4),
    }

    # Flight Leg Definition
    FlightLeg = namedtuple(
        "FlightLeg",
        "id depA depT arrA arrT dist pax price"
    )
    FlightLegs = IndexedSet([FlightLeg( 0, 'ABZ', 87, 'AMS', 337, 'Medium', 137, 3000),FlightLeg( 1, 'ABZ', 202, 'NWI', 452, 'Medium', 119, 1000),FlightLeg( 2, 'ABZ', 472, 'SVG', 683, 'Medium', 54, 1000),FlightLeg( 3, 'ABZ', 510, 'AMS', 760, 'Medium', 100, 3000),FlightLeg( 4, 'ABZ', 587, 'BGO', 799, 'Medium', 114, 1000),FlightLeg( 5, 'ABZ', 1165, 'SVG', 1376, 'Medium', 144, 1000),FlightLeg( 6, 'ABZ', 1511, 'NWI', 1761, 'Medium', 96, 1000),FlightLeg( 7, 'ABZ', 1550, 'AMS', 1761, 'Medium', 48, 3000),FlightLeg( 8, 'ABZ', 1781, 'SVG', 1992, 'Medium', 72, 1000),FlightLeg( 9, 'ABZ', 1858, 'AMS', 2069, 'Medium', 84, 3000),FlightLeg( 10, 'ABZ', 2127, 'NWI', 2377, 'Medium', 40, 1000),FlightLeg( 11, 'ABZ', 2204, 'SVG', 2416, 'Medium', 120, 1000),FlightLeg( 12, 'ABZ', 2628, 'AMS', 2878, 'Medium', 67, 3000),FlightLeg( 13, 'ABZ', 2743, 'NWI', 2993, 'Medium', 40, 1000),FlightLeg( 14, 'ABZ', 3013, 'SVG', 3224, 'Medium', 66, 1000),FlightLeg( 15, 'ABZ', 3051, 'AMS', 3301, 'Medium', 137, 3000),FlightLeg( 16, 'ABZ', 3128, 'BGO', 3340, 'Medium', 120, 1000),FlightLeg( 17, 'ABZ', 3706, 'SVG', 3917, 'Medium', 102, 1000),FlightLeg( 18, 'ABZ', 4052, 'NWI', 4302, 'Medium', 40, 1000),FlightLeg( 19, 'ABZ', 4091, 'AMS', 4302, 'Medium', 66, 3000),FlightLeg( 20, 'ABZ', 4322, 'SVG', 4533, 'Medium', 96, 1000),FlightLeg( 21, 'ABZ', 4399, 'AMS', 4610, 'Medium', 40, 3000),FlightLeg( 22, 'ABZ', 4668, 'NWI', 4918, 'Medium', 40, 1000),FlightLeg( 23, 'ABZ', 4745, 'SVG', 4957, 'Medium', 40, 1000),FlightLeg( 24, 'ABZ', 5130, 'AMS', 5380, 'Medium', 94, 3000),FlightLeg( 25, 'ABZ', 5554, 'AMS', 5804, 'Medium', 124, 3000),FlightLeg( 26, 'ABZ', 6593, 'AMS', 6766, 'Medium', 126, 3000),FlightLeg( 27, 'ABZ', 6863, 'AMS', 7036, 'Medium', 96, 3000),FlightLeg( 28, 'ABZ', 7440, 'AMS', 7613, 'Medium', 108, 3000),FlightLeg( 29, 'ABZ', 7748, 'AMS', 7998, 'Medium', 65, 3000),FlightLeg( 30, 'ABZ', 8403, 'SVG', 8614, 'Medium', 132, 1000),FlightLeg( 31, 'ABZ', 8788, 'BGO', 8999, 'Medium', 40, 1000),FlightLeg( 32, 'ABZ', 8788, 'AMS', 8999, 'Medium', 126, 3000),FlightLeg( 33, 'ABZ', 9019, 'SVG', 9230, 'Medium', 78, 1000),FlightLeg( 34, 'ABZ', 9096, 'AMS', 9307, 'Medium', 114, 3000),FlightLeg( 35, 'ABZ', 9365, 'NWI', 9615, 'Medium', 40, 1000),FlightLeg( 36, 'ABZ', 9442, 'SVG', 9654, 'Medium', 102, 1000),FlightLeg( 37, 'AMS', 87, 'BHX', 260, 'Short', 95, 1000),FlightLeg( 38, 'AMS', 125, 'ABZ', 375, 'Medium', 145, 3000),FlightLeg( 39, 'AMS', 356, 'EDI', 606, 'Medium', 108, 2000),FlightLeg( 40, 'AMS', 433, 'BHX', 606, 'Short', 115, 1000),FlightLeg( 41, 'AMS', 433, 'NWI', 568, 'Short', 144, 1000),FlightLeg( 42, 'AMS', 472, 'LBA', 645, 'Short', 40, 1000),FlightLeg( 43, 'AMS', 549, 'ABZ', 799, 'Medium', 81, 3000),FlightLeg( 44, 'AMS', 703, 'EDI', 914, 'Medium', 138, 2000),FlightLeg( 45, 'AMS', 1088, 'BHX', 1261, 'Short', 70, 1000),FlightLeg( 46, 'AMS', 1088, 'EDI', 1299, 'Medium', 96, 2000),FlightLeg( 47, 'AMS', 1088, 'NWI', 1222, 'Short', 94, 1000),FlightLeg( 48, 'AMS', 1165, 'LBA', 1338, 'Short', 115, 1000),FlightLeg( 49, 'AMS', 1165, 'ABZ', 1415, 'Medium', 76, 3000),FlightLeg( 50, 'AMS', 1627, 'EDI', 1838, 'Medium', 132, 2000),FlightLeg( 51, 'AMS', 1742, 'NWI', 1877, 'Short', 40, 1000),FlightLeg( 52, 'AMS', 1742, 'BHX', 1915, 'Short', 110, 1000),FlightLeg( 53, 'AMS', 1935, 'BHX', 2146, 'Medium', 42, 1000),FlightLeg( 54, 'AMS', 1935, 'EDI', 2185, 'Medium', 40, 2000),FlightLeg( 55, 'AMS', 1973, 'ABZ', 2223, 'Medium', 40, 3000),FlightLeg( 56, 'AMS', 1973, 'LBA', 2146, 'Short', 115, 1000),FlightLeg( 57, 'AMS', 2204, 'NWI', 2339, 'Short', 114, 1000),FlightLeg( 58, 'AMS', 2358, 'BHX', 2493, 'Short', 95, 1000),FlightLeg( 59, 'AMS', 2628, 'BHX', 2801, 'Short', 140, 1000),FlightLeg( 60, 'AMS', 2666, 'ABZ', 2916, 'Medium', 112, 3000),FlightLeg( 61, 'AMS', 2897, 'EDI', 3147, 'Medium', 95, 2000),FlightLeg( 62, 'AMS', 2974, 'BHX', 3147, 'Short', 100, 1000),FlightLeg( 63, 'AMS', 2974, 'NWI', 3109, 'Short', 40, 1000),FlightLeg( 64, 'AMS', 3013, 'LBA', 3186, 'Short', 90, 1000),FlightLeg( 65, 'AMS', 3090, 'ABZ', 3340, 'Medium', 142, 3000),FlightLeg( 66, 'AMS', 3244, 'EDI', 3455, 'Medium', 90, 2000),FlightLeg( 67, 'AMS', 3629, 'BHX', 3802, 'Short', 40, 1000),FlightLeg( 68, 'AMS', 3629, 'EDI', 3840, 'Medium', 84, 2000),FlightLeg( 69, 'AMS', 3629, 'NWI', 3763, 'Short', 142, 1000),FlightLeg( 70, 'AMS', 3706, 'LBA', 3879, 'Short', 130, 1000),FlightLeg( 71, 'AMS', 3706, 'ABZ', 3956, 'Medium', 135, 3000),FlightLeg( 72, 'AMS', 4168, 'EDI', 4379, 'Medium', 90, 2000),FlightLeg( 73, 'AMS', 4283, 'NWI', 4418, 'Short', 66, 1000),FlightLeg( 74, 'AMS', 4283, 'BHX', 4456, 'Short', 50, 1000),FlightLeg( 75, 'AMS', 4476, 'BHX', 4687, 'Medium', 40, 1000),FlightLeg( 76, 'AMS', 4476, 'EDI', 4726, 'Medium', 42, 2000),FlightLeg( 77, 'AMS', 4514, 'ABZ', 4764, 'Medium', 57, 3000),FlightLeg( 78, 'AMS', 4514, 'LBA', 4687, 'Short', 40, 1000),FlightLeg( 79, 'AMS', 4745, 'NWI', 4880, 'Short', 92, 1000),FlightLeg( 80, 'AMS', 4899, 'BHX', 5034, 'Short', 45, 1000),FlightLeg( 81, 'AMS', 5169, 'ABZ', 5419, 'Medium', 40, 3000),FlightLeg( 82, 'AMS', 5400, 'EDI', 5650, 'Medium', 138, 2000),FlightLeg( 83, 'AMS', 5477, 'BHX', 5650, 'Short', 40, 1000),FlightLeg( 84, 'AMS', 5477, 'NWI', 5611, 'Short', 40, 1000),FlightLeg( 85, 'AMS', 5515, 'LBA', 5688, 'Short', 100, 1000),FlightLeg( 86, 'AMS', 5592, 'ABZ', 5842, 'Medium', 112, 3000),FlightLeg( 87, 'AMS', 6131, 'BHX', 6304, 'Short', 130, 1000),FlightLeg( 88, 'AMS', 6131, 'EDI', 6343, 'Medium', 40, 2000),FlightLeg( 89, 'AMS', 6131, 'NWI', 6266, 'Short', 82, 1000),FlightLeg( 90, 'AMS', 6208, 'LBA', 6381, 'Short', 120, 1000),FlightLeg( 91, 'AMS', 6208, 'ABZ', 6458, 'Medium', 123, 3000),FlightLeg( 92, 'AMS', 6478, 'EDI', 6651, 'Medium', 40, 2000),FlightLeg( 93, 'AMS', 6940, 'BHX', 7113, 'Medium', 138, 1000),FlightLeg( 94, 'AMS', 6940, 'EDI', 7151, 'Medium', 59, 2000),FlightLeg( 95, 'AMS', 6978, 'NWI', 7074, 'Short', 52, 1000),FlightLeg( 96, 'AMS', 6978, 'ABZ', 7190, 'Medium', 40, 3000),FlightLeg( 97, 'AMS', 6978, 'LBA', 7113, 'Short', 60, 1000),FlightLeg( 98, 'AMS', 7363, 'BHX', 7536, 'Short', 45, 1000),FlightLeg( 99, 'AMS', 7402, 'ABZ', 7613, 'Medium', 137, 3000),FlightLeg( 100, 'AMS', 7671, 'BHX', 7844, 'Short', 40, 1000),FlightLeg( 101, 'AMS', 7671, 'NWI', 7806, 'Short', 110, 1000),FlightLeg( 102, 'AMS', 7710, 'EDI', 7921, 'Medium', 84, 2000),FlightLeg( 103, 'AMS', 7710, 'LBA', 7883, 'Short', 90, 1000),FlightLeg( 104, 'AMS', 7787, 'ABZ', 8037, 'Medium', 78, 3000),FlightLeg( 105, 'AMS', 8326, 'BHX', 8499, 'Short', 40, 1000),FlightLeg( 106, 'AMS', 8326, 'EDI', 8537, 'Medium', 40, 2000),FlightLeg( 107, 'AMS', 8326, 'NWI', 8460, 'Short', 74, 1000),FlightLeg( 108, 'AMS', 8403, 'LBA', 8576, 'Short', 115, 1000),FlightLeg( 109, 'AMS', 8403, 'ABZ', 8653, 'Medium', 102, 3000),FlightLeg( 110, 'AMS', 8865, 'EDI', 9076, 'Medium', 126, 2000),FlightLeg( 111, 'AMS', 9173, 'BHX', 9384, 'Medium', 120, 1000),FlightLeg( 112, 'AMS', 9173, 'EDI', 9423, 'Medium', 68, 2000),FlightLeg( 113, 'AMS', 9211, 'NWI', 9346, 'Short', 88, 1000),FlightLeg( 114, 'AMS', 9211, 'ABZ', 9461, 'Medium', 90, 3000),FlightLeg( 115, 'AMS', 9211, 'LBA', 9384, 'Short', 45, 1000),FlightLeg( 116, 'AMS', 9596, 'BHX', 9731, 'Short', 85, 1000),FlightLeg( 117, 'BGO', 895, 'ABZ', 1145, 'Medium', 111, 1000),FlightLeg( 118, 'BGO', 3436, 'ABZ', 3686, 'Medium', 129, 1000),FlightLeg( 119, 'BGO', 9096, 'ABZ', 9346, 'Medium', 84, 1000),FlightLeg( 120, 'BHX', 48, 'AMS', 221, 'Short', 80, 1000),FlightLeg( 121, 'BHX', 395, 'AMS', 568, 'Short', 40, 1000),FlightLeg( 122, 'BHX', 741, 'AMS', 914, 'Short', 55, 1000),FlightLeg( 123, 'BHX', 1434, 'AMS', 1607, 'Short', 40, 1000),FlightLeg( 124, 'BHX', 2050, 'AMS', 2223, 'Short', 40, 1000),FlightLeg( 125, 'BHX', 2243, 'AMS', 2416, 'Short', 85, 1000),FlightLeg( 126, 'BHX', 2589, 'AMS', 2762, 'Short', 75, 1000),FlightLeg( 127, 'BHX', 2936, 'AMS', 3109, 'Short', 40, 1000),FlightLeg( 128, 'BHX', 3282, 'AMS', 3455, 'Short', 55, 1000),FlightLeg( 129, 'BHX', 3975, 'AMS', 4148, 'Short', 110, 1000),FlightLeg( 130, 'BHX', 4591, 'AMS', 4764, 'Short', 100, 1000),FlightLeg( 131, 'BHX', 4784, 'AMS', 4957, 'Short', 50, 1000),FlightLeg( 132, 'BHX', 5092, 'AMS', 5265, 'Short', 105, 1000),FlightLeg( 133, 'BHX', 5785, 'AMS', 5958, 'Short', 40, 1000),FlightLeg( 134, 'BHX', 6478, 'AMS', 6612, 'Short', 80, 1000),FlightLeg( 135, 'BHX', 7325, 'AMS', 7498, 'Short', 50, 1000),FlightLeg( 136, 'BHX', 7633, 'AMS', 7806, 'Short', 135, 1000),FlightLeg( 137, 'BHX', 7979, 'AMS', 8152, 'Short', 55, 1000),FlightLeg( 138, 'BHX', 8672, 'AMS', 8845, 'Short', 40, 1000),FlightLeg( 139, 'BHX', 9481, 'AMS', 9654, 'Short', 135, 1000),FlightLeg( 140, 'DUB', 318, 'EMA', 452, 'Short', 40, 1000),FlightLeg( 141, 'DUB', 780, 'EMA', 914, 'Short', 98, 1000),FlightLeg( 142, 'DUB', 1357, 'EMA', 1492, 'Short', 40, 1000),FlightLeg( 143, 'DUB', 1819, 'EMA', 1954, 'Short', 132, 1000),FlightLeg( 144, 'DUB', 2859, 'EMA', 2993, 'Short', 80, 1000),FlightLeg( 145, 'DUB', 3321, 'EMA', 3455, 'Short', 86, 1000),FlightLeg( 146, 'DUB', 3898, 'EMA', 4033, 'Short', 136, 1000),FlightLeg( 147, 'DUB', 4360, 'EMA', 4495, 'Short', 40, 1000),FlightLeg( 148, 'DUB', 5477, 'EMA', 5611, 'Short', 134, 1000),FlightLeg( 149, 'DUB', 5939, 'EMA', 6073, 'Short', 102, 1000),FlightLeg( 150, 'DUB', 6824, 'EMA', 6959, 'Short', 40, 1000),FlightLeg( 151, 'DUB', 8018, 'EMA', 8152, 'Short', 40, 1000),FlightLeg( 152, 'DUB', 8595, 'EMA', 8730, 'Short', 74, 1000),FlightLeg( 153, 'DUB', 9057, 'EMA', 9192, 'Short', 70, 1000),FlightLeg( 154, 'EMA', 549, 'DUB', 683, 'Short', 102, 1000),FlightLeg( 155, 'EMA', 1011, 'DUB', 1145, 'Short', 122, 1000),FlightLeg( 156, 'EMA', 1588, 'DUB', 1723, 'Short', 40, 1000),FlightLeg( 157, 'EMA', 2050, 'DUB', 2185, 'Short', 40, 1000),FlightLeg( 158, 'EMA', 3090, 'DUB', 3224, 'Short', 48, 1000),FlightLeg( 159, 'EMA', 3552, 'DUB', 3686, 'Short', 40, 1000),FlightLeg( 160, 'EMA', 4129, 'DUB', 4264, 'Short', 40, 1000),FlightLeg( 161, 'EMA', 4591, 'DUB', 4726, 'Short', 86, 1000),FlightLeg( 162, 'EMA', 5708, 'DUB', 5842, 'Short', 96, 1000),FlightLeg( 163, 'EMA', 6170, 'DUB', 6304, 'Short', 98, 1000),FlightLeg( 164, 'EMA', 7017, 'DUB', 7151, 'Short', 92, 1000),FlightLeg( 165, 'EMA', 8249, 'DUB', 8383, 'Short', 132, 1000),FlightLeg( 166, 'EMA', 8826, 'DUB', 8961, 'Short', 130, 1000),FlightLeg( 167, 'EMA', 9288, 'DUB', 9423, 'Short', 114, 1000),FlightLeg( 168, 'EDI', 87, 'AMS', 298, 'Medium', 40, 2000),FlightLeg( 169, 'EDI', 703, 'AMS', 953, 'Medium', 50, 2000),FlightLeg( 170, 'EDI', 1049, 'AMS', 1261, 'Medium', 96, 2000),FlightLeg( 171, 'EDI', 1511, 'AMS', 1723, 'Medium', 66, 2000),FlightLeg( 172, 'EDI', 1973, 'AMS', 2185, 'Medium', 144, 2000),FlightLeg( 173, 'EDI', 2628, 'AMS', 2839, 'Medium', 120, 2000),FlightLeg( 174, 'EDI', 3244, 'AMS', 3494, 'Medium', 72, 2000),FlightLeg( 175, 'EDI', 3590, 'AMS', 3802, 'Medium', 126, 2000),FlightLeg( 176, 'EDI', 4052, 'AMS', 4264, 'Medium', 138, 2000),FlightLeg( 177, 'EDI', 4514, 'AMS', 4726, 'Medium', 144, 2000),FlightLeg( 178, 'EDI', 5130, 'AMS', 5342, 'Medium', 108, 2000),FlightLeg( 179, 'EDI', 5746, 'AMS', 5996, 'Medium', 88, 2000),FlightLeg( 180, 'EDI', 6555, 'AMS', 6728, 'Medium', 40, 2000),FlightLeg( 181, 'EDI', 6824, 'AMS', 6997, 'Medium', 54, 2000),FlightLeg( 182, 'EDI', 7363, 'AMS', 7575, 'Medium', 40, 2000),FlightLeg( 183, 'EDI', 7941, 'AMS', 8191, 'Medium', 43, 2000),FlightLeg( 184, 'EDI', 8749, 'AMS', 8961, 'Medium', 108, 2000),FlightLeg( 185, 'EDI', 9211, 'AMS', 9423, 'Medium', 78, 2000),FlightLeg( 186, 'LBA', 48, 'AMS', 221, 'Short', 40, 1000),FlightLeg( 187, 'LBA', 741, 'AMS', 914, 'Short', 40, 1000),FlightLeg( 188, 'LBA', 1473, 'AMS', 1646, 'Short', 40, 1000),FlightLeg( 189, 'LBA', 2589, 'AMS', 2762, 'Short', 85, 1000),FlightLeg( 190, 'LBA', 3282, 'AMS', 3455, 'Short', 40, 1000),FlightLeg( 191, 'LBA', 4014, 'AMS', 4187, 'Short', 65, 1000),FlightLeg( 192, 'LBA', 5092, 'AMS', 5265, 'Short', 90, 1000),FlightLeg( 193, 'LBA', 5785, 'AMS', 5958, 'Short', 65, 1000),FlightLeg( 194, 'LBA', 6516, 'AMS', 6651, 'Short', 85, 1000),FlightLeg( 195, 'LBA', 7325, 'AMS', 7498, 'Short', 105, 1000),FlightLeg( 196, 'LBA', 7979, 'AMS', 8152, 'Short', 40, 1000),FlightLeg( 197, 'LBA', 8711, 'AMS', 8884, 'Short', 85, 1000),FlightLeg( 198, 'NWI', 87, 'AMS', 221, 'Short', 52, 1000),FlightLeg( 199, 'NWI', 125, 'ABZ', 375, 'Medium', 148, 1000),FlightLeg( 200, 'NWI', 664, 'AMS', 799, 'Short', 86, 1000),FlightLeg( 201, 'NWI', 1473, 'AMS', 1607, 'Short', 40, 1000),FlightLeg( 202, 'NWI', 1511, 'ABZ', 1761, 'Medium', 95, 1000),FlightLeg( 203, 'NWI', 1858, 'ABZ', 2108, 'Medium', 57, 1000),FlightLeg( 204, 'NWI', 1973, 'AMS', 2108, 'Short', 50, 1000),FlightLeg( 205, 'NWI', 2628, 'AMS', 2762, 'Short', 146, 1000),FlightLeg( 206, 'NWI', 2666, 'ABZ', 2916, 'Medium', 133, 1000),FlightLeg( 207, 'NWI', 3205, 'AMS', 3340, 'Short', 74, 1000),FlightLeg( 208, 'NWI', 4014, 'AMS', 4148, 'Short', 142, 1000),FlightLeg( 209, 'NWI', 4052, 'ABZ', 4302, 'Medium', 40, 1000),FlightLeg( 210, 'NWI', 4399, 'ABZ', 4649, 'Medium', 68, 1000),FlightLeg( 211, 'NWI', 4514, 'AMS', 4649, 'Short', 82, 1000),FlightLeg( 212, 'NWI', 5130, 'AMS', 5265, 'Short', 126, 1000),FlightLeg( 213, 'NWI', 5708, 'AMS', 5842, 'Short', 86, 1000),FlightLeg( 214, 'NWI', 6516, 'AMS', 6612, 'Short', 106, 1000),FlightLeg( 215, 'NWI', 7363, 'AMS', 7498, 'Short', 40, 1000),FlightLeg( 216, 'NWI', 7902, 'AMS', 8037, 'Short', 40, 1000),FlightLeg( 217, 'NWI', 8711, 'AMS', 8845, 'Short', 40, 1000),FlightLeg( 218, 'NWI', 8903, 'ABZ', 9153, 'Medium', 51, 1000),FlightLeg( 219, 'SVG', 241, 'ABZ', 452, 'Medium', 108, 1000),FlightLeg( 220, 'SVG', 780, 'ABZ', 991, 'Medium', 138, 1000),FlightLeg( 221, 'SVG', 1473, 'ABZ', 1684, 'Medium', 72, 1000),FlightLeg( 222, 'SVG', 2089, 'ABZ', 2300, 'Medium', 84, 1000),FlightLeg( 223, 'SVG', 2782, 'ABZ', 2993, 'Medium', 40, 1000),FlightLeg( 224, 'SVG', 3321, 'ABZ', 3532, 'Medium', 138, 1000),FlightLeg( 225, 'SVG', 4014, 'ABZ', 4225, 'Medium', 40, 1000),FlightLeg( 226, 'SVG', 4630, 'ABZ', 4841, 'Medium', 48, 1000),FlightLeg( 227, 'SVG', 5631, 'ABZ', 5842, 'Medium', 108, 1000),FlightLeg( 228, 'SVG', 8711, 'ABZ', 8922, 'Medium', 40, 1000),FlightLeg( 229, 'SVG', 9327, 'ABZ', 9538, 'Medium', 126, 1000)
                ])
    OneStopFlights = IndexedSet([(0, 41)])

    # Source and Sink Flights
    Source = IndexedSet(
        [
            FlightLeg(
                999,
                "Source",
                f.depT - MaxRefuel,
                f.depA,
                f.depT - MaxRefuel,
                f.dist,
                0,
                0,
            )
            for f in FlightLegs
        ]
    )

    Sink = IndexedSet(
        [
            FlightLeg(
                999,
                f.arrA,
                Horizon + MaxRefuel,
                "Sink",
                Horizon + MaxRefuel,
                f.dist,
                0,
                0,
            )
            for f in FlightLegs
        ]
    )

    Flights = FlightLegs | Source | Sink

    # Cost and Profit
    Cost = {fl: {pl: 0 for pl in Fleets} for fl in Flights}
    Profit = {fl: {pl: 0 for pl in Fleets} for fl in Flights}

    for fl in Flights:
        for pl in Fleets:

            if fl.depA != "Source" and fl.arrA != "Sink":

                factor = min(FleetInfo[pl].seats, fl.pax)

                Cost[fl][pl] = (
                    FleetInfo[pl].a
                    + FleetInfo[pl].b * factor * (fl.arrT - fl.depT)
                )

                Profit[fl][pl] = factor * fl.price

    # Build OPL Model
    mdl = ModelReader.build_opl_model(
        mod,
        Airports=Airports,
        MaxSpill=MaxSpill,
        Fleets=Fleets,
        FleetInfo=FleetInfo,
        MaxRefuel=MaxRefuel,
        Horizon=Horizon,
        FlightLegs=FlightLegs,
        OneStopFlights=OneStopFlights,
        Flights=Flights,
        Cost=Cost,
        Profit=Profit,
    )

    # Solve Model
    sol = mdl.solve()

    if sol:
        print("Objective Value:", sol.get_objective_value())
    else:
        print("No solution found")

    # Extract Variables
    all_Var = mdl.get_opl_var()
    print("OPL Variables:", all_Var.variable_names)

    # Access variable
    all_Var.Assignment

    # Extract solution
    all_Var_sol = all_Var.from_solution(sol)

    print("\nSolutions:")

    for i in all_Var_sol.Assignment:
        for j in all_Var_sol.Assignment[i]:
            print(all_Var.Assignment[i][j], ":", all_Var_sol.Assignment[i][j])


if __name__ == "__main__":
    main()