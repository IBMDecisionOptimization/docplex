# --------------------------------------------------------------------------
# Version 22.2.0
# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2000, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
"""Enum constants from the CPLEX C Callable Library"""

CPXCALLBACKINFO_THREADID = 0
CPXCALLBACKINFO_NODECOUNT = 1
CPXCALLBACKINFO_ITCOUNT = 2
CPXCALLBACKINFO_BEST_SOL = 3
CPXCALLBACKINFO_BEST_BND = 4
CPXCALLBACKINFO_THREADS = 5
CPXCALLBACKINFO_FEASIBLE = 6
CPXCALLBACKINFO_TIME = 7
CPXCALLBACKINFO_DETTIME = 8
CPXCALLBACKINFO_NODEUID = 9
CPXCALLBACKINFO_NODEDEPTH = 10
CPXCALLBACKINFO_CANDIDATE_SOURCE = 11
CPXCALLBACKINFO_RESTARTS = 12
CPXCALLBACKINFO_AFTERCUTLOOP = 13
CPXCALLBACKINFO_NODESLEFT = 14
CPXCALLBACKSOLUTION_NOCHECK = -1
CPXCALLBACKSOLUTION_CHECKFEAS = 0
CPXCALLBACKSOLUTION_PROPAGATE = 1
CPXCALLBACKSOLUTION_SOLVE = 2
