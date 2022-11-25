# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The RCPSP (Resource-Constrained Project Scheduling Problem) is a generalization
of the production-specific Job-Shop (see job_shop_basic.py), Flow-Shop
(see flow_shop.py) and Open-Shop(see open_shop.py) scheduling problems.

Given:
- a set of q resources with given capacities,
- a network of precedence constraints between the activities, and
- for each activity and each resource the amount of the resource
  required by the activity over its execution,
the goal of the RCPSP is to find a schedule meeting all the
constraints whose makespan (i.e., the time at which all activities are
finished) is minimal.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *
import os

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Read the input data file.
# Available files are rcpsp_default, and different rcpsp_XXXXXX.
# First line contains the number of tasks, and the number of resources.
# Second line contains the capacities of the resources.
# The rest of the file consists of one line per task, organized as follows:
# - duration of the task
# - the demand on each resource (one integer per resource)
# - the number of successors followed by the list of successor numbers

filename = os.path.dirname(os.path.abspath(__file__)) + '/data/rcpsp_default.data'
with open(filename, 'r') as file:
    NB_TASKS, NB_RESOURCES = [int(v) for v in file.readline().split()]
    CAPACITIES = [int(v) for v in file.readline().split()]
    TASKS = [[int(v) for v in file.readline().split()] for i in range(NB_TASKS)]


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Extract duration of each task
DURATIONS = [TASKS[t][0] for t in range(NB_TASKS)]

# Extract demand of each task
DEMANDS = [TASKS[t][1:NB_RESOURCES+1] for t in range(NB_TASKS)]

# Extract successors of each task
SUCCESSORS = [TASKS[t][NB_RESOURCES+2:] for t in range(NB_TASKS)]


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create task interval variables
tasks = [interval_var(name='T{}'.format(i+1), size=DURATIONS[i]) for i in range(NB_TASKS)]

# Add precedence constraints
mdl.add(end_before_start(tasks[t], tasks[s-1]) for t in range(NB_TASKS) for s in SUCCESSORS[t])

# Constrain capacity of resources
mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r] for r in range(NB_RESOURCES))

# Minimize end of all tasks
mdl.add(minimize(max(end_of(t) for t in tasks)))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=100000,TimeLimit=10)
print('Solution: ')
res.print_solution()

import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    load = [CpoStepFunction() for j in range(NB_RESOURCES)]
    for i in range(NB_TASKS):
        itv = res.get_var_solution(tasks[i])
        for j in range(NB_RESOURCES):
            if 0 < DEMANDS[i][j]:
                load[j].add_value(itv.get_start(), itv.get_end(), DEMANDS[i][j])

    visu.timeline('Solution for RCPSP ' + filename)
    visu.panel('Tasks')
    for i in range(NB_TASKS):
        visu.interval(res.get_var_solution(tasks[i]), i, tasks[i].get_name())
    for j in range(NB_RESOURCES):
        visu.panel('R' + str(j+1))
        visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, CAPACITIES[j])], style='area', color='lightgrey')
        visu.function(segments=load[j], style='area', color=j)
    visu.show()
