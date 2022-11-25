# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The general Flow-Shop scheduling problem is a production problem where
a set of n jobs have to be processed with identical flow pattern on m
machines (see flow_shop.py).

In permutation flow-shops the sequence of jobs is the same on all machines.
For example, if machine 1 runs jobs J2, J1, J5, then the order is the same
on all other machines.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *
import os


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Read the input data file.
# Available files are flowshop_default, and different flowshop_tailXXXX.
# First line contains the number of jobs, and the number of machines.
# The rest of the file consists of one line per job that contains the list of
# operations given as durations for each machines.
filename = os.path.dirname(os.path.abspath(__file__)) + '/data/flowshop_default.data'
with open(filename, 'r') as file:
    NB_JOBS, NB_MACHINES = [int(v) for v in file.readline().split()]
    OP_DURATIONS = [[int(v) for v in file.readline().split()] for i in range(NB_JOBS)]


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create one interval variable per job operation
operations = [[interval_var(size=OP_DURATIONS[j][m], name='J{}-M{}'.format(j, m)) for m in range(NB_MACHINES)] for j in range(NB_JOBS)]

# Create sequence of operation for each machine
op_sequences = [sequence_var([operations[i][j] for i in range(NB_JOBS)], name='M{}'.format(j)) for j in range(NB_MACHINES)]

# Force each operation to start after the end of the previous
for j in range(NB_JOBS):
    for m in range(1, NB_MACHINES):
        mdl.add(end_before_start(operations[j][m-1], operations[j][m]))

# Force no overlap for operations executed on a same machine
for m in range(NB_MACHINES):
    mdl.add(no_overlap(op_sequences[m]))

# Force sequences to be all identical on all machines
for m in range(1, NB_MACHINES):
    mdl.add(same_sequence(op_sequences[0], op_sequences[m]))

# Minimize termination date
mdl.add(minimize(max([end_of(operations[i][NB_MACHINES-1]) for i in range(NB_JOBS)])))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=10000, TimeLimit=10)

# Draw solution
import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    visu.timeline('Solution for permutation flow-shop ' + filename)
    visu.panel('Jobs')
    for i in range(NB_JOBS):
        visu.sequence(name='J' + str(i),
                      intervals=[(res.get_var_solution(operations[i][j]), j, 'M' + str(j)) for j in range(NB_MACHINES)])
    visu.panel('Machines')
    for j in range(NB_MACHINES):
        visu.sequence(name='M' + str(j),
                      intervals=[(res.get_var_solution(operations[i][j]), j, 'J' + str(i)) for i in range(NB_JOBS)])
    visu.show()
