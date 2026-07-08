# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Test that CPLEX reports the correct number of cores."""
import os
import sys
import cplex

def numcores():
    # For Python >= 2.6 we could just use multiprocessing.cpu_count()
    # On POSIX systems we use sysconf() to query the number
    # of processors online.
    try:
        cores = int(os.sysconf('SC_NPROCESSORS_ONLN'))
        if cores > 0:
            return cores
    except (AttributeError, ValueError):
        pass

    # On Windows we can read the processor count from the environment.
    try:
        cores = int(os.environ['NUMBER_OF_PROCESSORS'])
        if cores > 0:
            return cores
    except:
        pass

    raise Exception("Could not figure out number of cores")

if __name__ == "__main__":
    c = cplex.Cplex()
    cpx = c.get_num_cores()
    builtin = numcores()

    if cpx != builtin:
        print("Unexpected number of cores ",cpx," expected(",builtin,")")
        raise Exception("Unexpected number of cores")
    else:
        print("Test passed (",cpx," cores)")
    del c
