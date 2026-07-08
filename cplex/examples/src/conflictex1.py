#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: conflictex1.py
 
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Reading a problem and using the conflict refiner.

See the usage message for more details.
"""
import sys
from collections import defaultdict
import cplex
from cplex.exceptions import CplexSolverError, error_codes


def usage(name):
    """Prints a usage statement."""
    msg = """\
Usage: {0} filename
  filename   Name of a file, with .mps, .lp, or .sav
             extension, and a possible, additional .gz
             extension
"""
    print(msg.format(name))
    sys.exit(2)


def main(filename):
    """Invoke the conflict refiner on a model and display the results."""
    c = cplex.Cplex()
    c.read(filename)

    # Invoke the conflict refiner and print the status. As opposed to
    # letting the conflict refiner run to completion, the user can
    # interrupt the conflict refiner with a Ctrl+C and still potentially
    # get a "possible" conflict.
    c.conflict.refine(c.conflict.all_constraints())
    print("Solution status: {0} ({1})".format(c.solution.get_status_string(),
                                              c.solution.get_status()))

    # Get the conflict status for the conflict group that was specified
    # (e.g., the all constraints group was used above).
    try:
        confstatus = c.conflict.get()
    except CplexSolverError as err:
        # If the conflict refiner was unable to identify a conflict then
        # exit early.
        if err.args[2] == error_codes.CPXERR_NO_CONFLICT:
            print("A conflict was not identified.")
            print("Exiting....")
            return
        else:
            raise

    # Get the expanded groups of constraints used by the conflict
    # refiner. That is, the groups specifieid in the last call to
    # conflict.refine(). Since we used the all constraints group above,
    # this will contain an entry for every upper bound, lower bound,
    # linear constraint, quadratic constraint, SOS constraint, indicator
    # constraint, and PWL constraint in the model.
    #
    # conflict.get_groups() returns a list of constraint groups. The
    # constraint groups are tuples of length two, the first entry of
    # which is the preference for the group (a float), the second of
    # which is a tuple of pairs (type, id), where type is an attribute
    # of conflict.constraint_type and id is an index for the type
    # (e.g., a linear constraint index).
    groups = c.conflict.get_groups()

    # For the purposes of this example, we only care about the constraint
    # types as they align with the conflict statuses.
    conftypes = []
    for pref, subgroup in groups:
        for grptype, index in subgroup:
            conftypes.append(grptype)

    # Count the number of conflicts found for each constraint group and
    # print the results.
    grpdict = defaultdict(int)
    for grptype, grpstat in zip(conftypes, confstatus):
        if grpstat in (c.conflict.group_status.possible_member,
                       c.conflict.group_status.member):
            grpdict[grptype] += 1

    for grptype in grpdict:
        print("{0}(s): {1}".format(c.conflict.constraint_type[grptype],
                                   grpdict[grptype]))

    # Write the identified conflict in the LP format.
    conffile = "conflictex1.py.lp"
    print("Writing conflict file to '{0}'....".format(conffile))
    c.conflict.write(conffile)

    # Display the entire conflict subproblem.
    with open(conffile, "r") as conf_file:
        for line in conf_file:
            print(line.strip())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage(sys.argv[0])
    main(sys.argv[1])
