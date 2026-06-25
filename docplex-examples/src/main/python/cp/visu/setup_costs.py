# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The problem is to schedule a set of tasks on two alternative machines with
different setup times.

The objective is to minimize the number of "long" setup times on machines.
A setup time is considered to be long if it is larger than 30.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Number of types
NB_TYPES = 10

# Setup times of machine M1
SETUP_M1 = [
    [22, 24, 7, 10, 9, 41, 14, 30, 24, 6],
    [63, 21, 42, 1, 24, 17, 35, 25, 0, 68],
    [60, 70, 37, 70, 39, 84, 44, 60, 67, 36],
    [77, 57, 65, 33, 81, 74, 72, 82, 57, 83],
    [51, 31, 18, 32, 48, 45, 51, 21, 28, 45],
    [46, 42, 29, 11, 11, 21, 59, 8, 4, 51],
    [35, 59, 42, 45, 44, 76, 37, 65, 59, 41],
    [38, 62, 45, 14, 33, 24, 52, 32, 7, 44],
    [63, 57, 44, 7, 26, 17, 55, 25, 21, 68],
    [24, 34, 1, 34, 3, 48, 8, 24, 31, 30]
]

# Setup times of machine M2
SETUP_M2 = [
    [27, 48, 44, 52, 21, 61, 33, 5, 37, 64],
    [42, 44, 42, 40, 17, 40, 49, 41, 66, 29],
    [36, 53, 31, 56, 50, 56, 7, 41, 49, 60],
    [6, 43, 46, 38, 16, 44, 39, 11, 43, 12],
    [25, 27, 45, 67, 37, 67, 52, 30, 62, 56],
    [6, 43, 2, 0, 16, 35, 9, 11, 43, 12],
    [29, 70, 25, 62, 43, 62, 26, 34, 42, 61],
    [22, 43, 53, 47, 16, 56, 28, 10, 32, 59],
    [56, 93, 73, 76, 66, 82, 48, 61, 51, 50],
    [18, 55, 34, 26, 28, 32, 40, 12, 44, 25]
]

# Task duration
TASK_DURATION = [
    19, 18, 16, 11, 16, 15, 19, 18, 17, 17,
    20, 16, 16, 14, 19, 11, 10, 16, 12, 20,
    14, 14, 20, 12, 18, 16, 10, 15, 11, 13,
    15, 11, 11, 13, 19, 17, 11, 20, 19, 17,
    15, 19, 13, 16, 20, 13, 13, 13, 13, 15
]

# Number of tasks
NB_TASKS = len(TASK_DURATION)

# Task type
TASK_TYPE = [
    8, 1, 6, 3, 4, 8, 8, 4, 3, 5,
    9, 4, 1, 5, 8, 8, 4, 1, 9, 2,
    6, 0, 8, 9, 1, 0, 1, 7, 5, 9,
    3, 1, 9, 3, 0, 7, 0, 7, 1, 4,
    5, 7, 4, 0, 9, 1, 5, 4, 5, 1
]


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Build tasks for machine M1 and M2
tasks_m1 = [interval_var(name='A{}_M1_TP{}'.format(i, TASK_TYPE[i]), optional=True) for i in range(NB_TASKS)]
tasks_m2 = [interval_var(name='A{}_M2_TP{}'.format(i, TASK_TYPE[i]), optional=True) for i in range(NB_TASKS)]

# Build actual tasks as an alternative between machines
tasks = [interval_var(name='A{}_TP{}'.format(i, TASK_TYPE[i]), size=TASK_DURATION[i]) for i in range(NB_TASKS)]
mdl.add(alternative(tasks[i], [tasks_m1[i], tasks_m2[i]]) for i in range(NB_TASKS))

# Build a map to retrieve task id from variable name (for display purpose)
task_id = dict()
for i in range(NB_TASKS):
    task_id[tasks_m1[i].get_name()] = i
    task_id[tasks_m2[i].get_name()] = i

# Constrain tasks to no overlap on each machine
s1 = sequence_var(tasks_m1, types=TASK_TYPE, name='M1')
s2 = sequence_var(tasks_m2, types=TASK_TYPE, name='M2')
mdl.add(no_overlap(s1, SETUP_M1, 1))
mdl.add(no_overlap(s2, SETUP_M2, 1))

# Minimize the number of 'long' setup times on machines.
nbLongSetups = 0
for i in range(NB_TASKS):
    tpi = TASK_TYPE[i]
    isLongSetup1 = [1 if 30 <= SETUP_M1[tpi][j] else 0 for j in range(NB_TYPES)] + [0]
    isLongSetup2 = [1 if 30 <= SETUP_M2[tpi][j] else 0 for j in range(NB_TYPES)] + [0]
    nbLongSetups += element(type_of_next(s1, tasks_m1[i], NB_TYPES, NB_TYPES), isLongSetup1)
    nbLongSetups += element(type_of_next(s2, tasks_m2[i], NB_TYPES, NB_TYPES), isLongSetup2)

mdl.add(minimize(nbLongSetups))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def compact(name):
    # Example: A31_M1_TP1 -> 31
    task, foo = name.split('_', 1)
    return task[1:]

# Solve model
print('Solving model...')
res = mdl.solve(TimeLimit=10, FailLimit=1000000)
print('Solution: ')
res.print_solution()

import docplex.cp.utils_visu as visu

def showsequence(s, setup):
    seq = res.get_var_solution(s)
    visu.sequence(name=s.get_name())
    vs = seq.get_value()
    for v in vs:
        nm = v.get_name()
        visu.interval(v, TASK_TYPE[task_id[nm]], compact(nm))
    for i in range(len(vs) - 1):
        end = vs[i].get_end()
        tp1 = TASK_TYPE[task_id[vs[i].get_name()]]
        tp2 = TASK_TYPE[task_id[vs[i + 1].get_name()]]
        visu.transition(end, end + setup[tp1][tp2])

if res and visu.is_visu_enabled():
    visu.timeline('Solution for transition costs')
    showsequence(s1, SETUP_M1)
    showsequence(s2, SETUP_M2)
    visu.show()
