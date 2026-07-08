# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This example is the same than the one implemented in rcpsp_multi_mode.py except that
input data files are represented with JSON format, simpler to read and modify.

The MMRCPSP (Multi-Mode Resource-Constrained Project Scheduling Problem) is a
generalization of the Resource-Constrained Project Scheduling problem
(see rcpsp.py).

In the MMRCPSP, each activity can be performed in one out of several modes.
Each mode of an activity represents an alternative way of combining different levels
of resource requirements with a related duration.
Renewable and non-renewable resources are distinguished.
While renewable resources have a limited instantaneous availability such as
manpower and machines, non renewable resources are limited for the entire project,
allowing to model, e.g., a budget for the project.

The objective is to find a mode and a start time for each activity such that the
schedule is makespan minimal and feasible with regard to the precedence
and resource constraints.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *
import os
import json


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Load input data from json file
filename = os.path.dirname(os.path.abspath(__file__)) + '/data/rcpspmm_default.json'
with open(filename, 'r') as f:
    jstr = f.read()
JSON_DATA = json.loads(jstr)


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Get renewable capacities
CAPACITIES_RENEWABLE = JSON_DATA['capacityRenewable']
NB_RENEWABLE = len(CAPACITIES_RENEWABLE)

# Get non-renewable capacities
CAPACITIES_NON_RENEWABLE = JSON_DATA['capacityNonRenewable']
NB_NON_RENEWABLE = len(CAPACITIES_NON_RENEWABLE)

# Get list of tasks
TASKS = JSON_DATA['tasks']
NB_TASKS = len(TASKS)

# Create a unique id for each mode (to retrieve results)
MODES = []
for t in TASKS:
   for i, m in enumerate(t['modes']):
       m['id'] = 'T{}-M{}'.format(t['id'], i + 1)
       MODES.append(m)


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create one interval variable per task
tasks = {t['id']: interval_var(name='T{}'.format(t['id'])) for t in TASKS}

# Add precedence constraints
mdl.add(end_before_start(tasks[t['id']], tasks[s]) for t in TASKS for s in t['successors'])

# Create one optional interval variable per task mode
modes = { m['id']: interval_var(name=m['id'], optional=True, size=m['duration']) for t in TASKS for m in t['modes'] }

# Add alternative constraints for tasks
mdl.add(alternative(tasks[t['id']], [ modes[m['id']] for m in t['modes'] ]) for t in TASKS)

# Initialize cumul functions for renewable and non renewable resources
renewables     = [ sum(pulse(modes[m['id']], m['demandRenewable'][j]) for m in MODES if m['demandRenewable'][j] > 0)
                   for j in range(NB_RENEWABLE)]
non_renewables = [ sum(m['demandNonRenewable'][j]*presence_of(modes[m['id']]) for m in MODES if m['demandNonRenewable'][j] > 0 )
                   for j in range(NB_NON_RENEWABLE)]

# Constrain renewable resources capacity
mdl.add(renewables[j] <= CAPACITIES_RENEWABLE[j] for j in range(NB_RENEWABLE))

# Constrain non-renewable resources capacity
mdl.add(non_renewables[j] <= CAPACITIES_NON_RENEWABLE[j] for j in range(NB_NON_RENEWABLE))

# Minimize overall schedule end date
mdl.add(minimize(max([end_of(t) for t in tasks.values()])))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=30000, TimeLimit=10)
print('Solution: ')
res.print_solution()

import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    load = [CpoStepFunction() for j in range(NB_RENEWABLE)]
    for m in MODES:
        itv = res.get_var_solution(modes[m['id']])
        if itv.is_present():
            for j in range(NB_RENEWABLE):
                dem = m['demandRenewable'][j]
                if dem > 0:
                    load[j].add_value(itv.get_start(), itv.get_end(), dem)

    visu.timeline('Solution for RCPSPMM ' + filename)
    visu.panel('Tasks')
    for t in TASKS:
        tid = t['id']
        visu.interval(res.get_var_solution(tasks[tid]), tid, str(tid))
    for j in range(NB_RENEWABLE):
        visu.panel('R' + str(j + 1))
        visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, CAPACITIES_RENEWABLE[j])], style='area', color='lightgrey')
        visu.function(segments=load[j], style='area', color=j)
    visu.show()
