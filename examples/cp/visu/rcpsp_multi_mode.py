# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
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

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Read next non empty line
def next_int_line(f):
    line = None
    while not line:
       line = f.readline().split()
    return [int(v) for v in line]

# Read the input data file.
# Available files are rcpspmm_default, and different rcpspmm_XXXXXX.
# First line contains the number of tasks, the number of renewable and non-renewable resources.
# Second line contains the capacities of the renewable resources.
# Third line contains the capacities of the non_renewable resources.
# The next nb_task lines are description of tasks, with:
# - the id of the task,
# - the number of modes for this task,
# - the number of successors followed by the list of successor numbers
# The rest of the file consists of one line per task mode, containing:
# - the id of the task (from zero)
# - the id of the mode (from 1)
# - the duration of the task in this mode
# - the demand for renewable resources
# - the demand for non-renewable resources

filename = os.path.dirname(os.path.abspath(__file__)) + '/data/rcpspmm_default.data'
with open(filename, 'r') as file:
    NB_TASKS, NB_RENEWABLE, NB_NON_RENEWABLE = next_int_line(file)
    CAPACITIES_RENEWABLE = next_int_line(file)
    CAPACITIES_NON_RENEWABLE = next_int_line(file)
    TASKS = [next_int_line(file) for i in range(NB_TASKS)]
    TASK_MODES = [next_int_line(file) for i in range(sum([t[1] for t in TASKS]))]


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Object class representing a task
class Task(object):
    def __init__(self, name, nb_modes):
        super(Task, self).__init__()
        self.name = name
        self.nb_modes = nb_modes
        self.successors = []
        self.modes = []

# Object class representing a mode
class Mode(object):
    def __init__(self, name, task, duration, dem_renewables, dem_non_renewables):
        super(Mode, self).__init__()
        self.name = name
        self.task = task
        self.duration = duration
        self.demand_renewable = dem_renewables
        self.demand_non_renewable = dem_non_renewables


# Build list of tasks
tasks_data = []
for i, t in enumerate(TASKS):
    task = Task('T{}'.format(i), t[1])
    for j in range(t[2]):
        task.successors.append(t[3 + j])
    tasks_data.append(task)

# Build list of modes
modes_data = []
for m in TASK_MODES:
    taskid = m[0]
    modeid = m[1]
    dur = m[2]
    dem_renewables = m[3 : 3 + NB_RENEWABLE]
    dem_non_renewables = m[3 + NB_RENEWABLE : 3 + NB_RENEWABLE + NB_NON_RENEWABLE]
    mode = Mode('T{}-M{}'.format(taskid, modeid),
                tasks_data[taskid], dur, dem_renewables, dem_non_renewables)
    tasks_data[taskid].modes.append(mode)
    modes_data.append(mode)


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create one interval variable per task
tasks = { t: mdl.interval_var(name=t.name) for t in tasks_data}

# Add precedence constraints
mdl.add(end_before_start(tasks[t], tasks[tasks_data[s]]) for t in tasks_data for s in t.successors)

# Create one optional interval variable per mode
modes = { m: interval_var(name=m.name, optional=True, size=m.duration) for m in modes_data}

# Add mode alternative for each task
mdl.add(alternative(tasks[t], [modes[m] for m in t.modes]) for t in tasks_data)

# Initialize cumul functions for renewable and non renewable resources
renewables     = [ sum(pulse(modes[m], m.demand_renewable[j]) for m in modes_data if m.demand_renewable[j] > 0)
                   for j in range(NB_RENEWABLE)]
non_renewables = [ sum(m.demand_non_renewable[j]*presence_of(modes[m]) for m in modes_data if m.demand_non_renewable[j] > 0 )
                   for j in range(NB_NON_RENEWABLE)]

# Constrain renewable resources capacity
mdl.add(renewables[j] <= CAPACITIES_RENEWABLE[j]  for j in range(NB_RENEWABLE))

# Constrain non-renewable resources capacity
mdl.add(non_renewables[j] <= CAPACITIES_NON_RENEWABLE[j]  for j in range(NB_NON_RENEWABLE))

# Minimize overall schedule end date
mdl.add(minimize(max([end_of(tasks[t]) for t in tasks_data])))


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
    for m in modes_data:
        itv = res.get_var_solution(modes[m])
        if itv.is_present():
            for j in range(NB_RENEWABLE):
                if 0 < m.demand_renewable[j]:
                    load[j].add_value(itv.get_start(), itv.get_end(), m.demand_renewable[j])

    visu.timeline('Solution for RCPSPMM ' + filename)
    visu.panel('Tasks')
    for t in tasks_data:
        visu.interval(res.get_var_solution(tasks[t]), int(t.name[1:]), t.name)
    for j in range(NB_RENEWABLE):
        visu.panel('R ' + str(j + 1))
        visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, CAPACITIES_RENEWABLE[j])], style='area', color='lightgrey')
        visu.function(segments=load[j], style='area', color=j)
    visu.show()
