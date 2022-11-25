# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This problem can be described as follows: a finite set of operations has to be
processed on a given set of machines.
Each operation has a specific processing time during which it may not be interrupted.
Operations are grouped in jobs, so that each operation belongs to exactly one job.
Furthermore, each operation requires exactly one machine for processing.

The objective of the problem is to schedule all operations, i.e., to determine
their start time, so as to minimize the maximum completion time (makespan)
given the additional constraints that: operations which belong to the same job and
operations which use the same machine cannot be processed simultaneously.

This problem is similar that the one proposed in flow_shop.py except that
job operations can be executed in any order.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *
import os


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Read the input data file.
# Available files are openshop_default, and different openshop_XXXX.
# First line contains the number of jobs, and the number of machines.
# The rest of the file consists of one line per job that contains the list of
# operations given as durations for each machines.
filename = os.path.dirname(os.path.abspath(__file__)) + '/data/openshop_default.data'
with open(filename, 'r') as file:
    NB_JOBS, NB_MACHINES = [int(v) for v in file.readline().split()]
    JOB_DURATIONS = [[int(v) for v in file.readline().split()] for i in range(NB_JOBS)]


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create one interval variable per job operation
job_operations = [[interval_var(size=JOB_DURATIONS[j][m], name='J{}-M{}'.format(j,m)) for m in range(NB_MACHINES)] for j in range(NB_JOBS)]

# All operations executed on the same machine must no overlap
mdl.add(no_overlap(job_operations[i][j] for j in range(NB_MACHINES)) for i in range(NB_JOBS))

# All operations executed for the same job must no overlap
mdl.add(no_overlap(job_operations[i][j] for i in range(NB_JOBS)) for j in range(NB_MACHINES))

# Minimization completion time
mdl.add(minimize(max(end_of(job_operations[i][j]) for i in range(NB_JOBS) for j in range(NB_MACHINES))))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=10000,TimeLimit=10)
print('Solution: ')
res.print_solution()

# Display solution
import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    visu.timeline('Solution for open-shop ' + filename)
    visu.panel('Jobs')
    for i in range(NB_JOBS):
        visu.sequence(name='J' + str(i),
                      intervals=[(res.get_var_solution(job_operations[i][j]), j, 'M' + str(j)) for j in range(NB_MACHINES)])
    visu.panel('Machines')
    for j in range(NB_MACHINES):
        visu.sequence(name='M' + str(j),
                      intervals=[(res.get_var_solution(job_operations[i][j]), j, 'J' + str(i)) for i in range(NB_JOBS)])
    visu.show()
