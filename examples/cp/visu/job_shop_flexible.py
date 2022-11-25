# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This problem is an extension of the classical Job-Shop Scheduling problem
(see job_shop_basic.py) which allows an operation to be processed by any machine
from a given set.
The operation processing time depends on the allocated machine.
The problem is to assign each operation to a machine and to order the operations
on the machines such that the maximal completion time (makespan) of all
operations is minimized.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *
import os


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Read the input data file.
# Available files are jobshopflex_default and multiple other jobshopflex_XXXX.
# First line contains the number of jobs, and the number of machines.
# The rest of the file consists of one line per job.
# First integer is the number of job steps, followed by the choices for each step.
# For each step, first integer indicates the number of choices, followed
# by the choices expressed with two integers: machine and duration
filename = os.path.dirname(os.path.abspath(__file__)) + '/data/jobshopflex_default.data'
with open(filename, 'r') as file:
    NB_JOBS, NB_MACHINES = [int(v) for v in file.readline().split()]
    list_jobs = [[int(v) for v in file.readline().split()] for i in range(NB_JOBS)]


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Build final list of jobs.
# Each job is a list of operations.
# Each operation is a list of choices expressed as tuples (machine, duration)
JOBS = []
for jline in list_jobs:
    nbstps = jline.pop(0)
    job = []
    for stp in range(nbstps):
        nbc = jline.pop(0)
        choices = []
        for c in range(nbc):
            m = jline.pop(0)
            d = jline.pop(0)
            choices.append((m - 1, d))
        job.append(choices)
    JOBS.append(job)


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Following code creates:
# - creates one interval variable 'ops' for each possible operation choice
# - creates one interval variable mops' for each operation, as an alternative of all operation choices
# - setup precedence constraints between operations of each job
# - creates a no_overlap constraint an the operations of each machine

ops  = { (j,o) : interval_var(name='J{}_O{}'.format(j,o))
         for j,J in enumerate(JOBS) for o,O in enumerate(J)}
mops = { (j,o,k,m) : interval_var(name='J{}_O{}_C{}_M{}'.format(j,o,k,m), optional=True, size=d)
         for j,J in enumerate(JOBS) for o,O in enumerate(J) for k, (m, d) in enumerate(O)}

# Precedence constraints between operations of a job
mdl.add(end_before_start(ops[j,o], ops[j,o-1]) for j,o in ops if 0<o)

# Alternative constraints
mdl.add(alternative(ops[j,o], [mops[a] for a in mops if a[0:2]==(j,o)]) for j,o in ops)

# Add no_overlap constraint between operations executed on the same machine
mdl.add(no_overlap(mops[a] for a in mops if a[3]==m) for m in range(NB_MACHINES))

# Minimize termination date
mdl.add(minimize(max(end_of(ops[j,o]) for j,o in ops)))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=100000,TimeLimit=10)
print('Solution:')
res.print_solution()

# Draw solution
import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
# Draw solution
    visu.timeline('Solution for flexible job-shop ' + filename)
    visu.panel('Machines')
    for m in range(NB_MACHINES):
        visu.sequence(name='M' + str(m))
        for a in mops:
            if a[3]==m:
                itv = res.get_var_solution(mops[a])
                if itv.is_present():
                    visu.interval(itv, a[0], 'J{}'.format(a[0]))
    visu.show()
