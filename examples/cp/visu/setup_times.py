# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This example solves a scheduling problem on two alternative heterogeneous machines.
A set of tasks {a_1,...,a_n} has to be executed on either one of the two machines.
Different types of tasks are distinguished, the type of task a_i is denoted tp_i.

A machine m needs a sequence dependent setup time setup(tp,tp') to switch from a
task of type tp to the next task of type tp'.
Furthermore some transitions tp->tp' are forbidden.

The two machines are different: they process tasks with different speed and have
different setup times and forbidden transitions.

The objective is to minimize the makespan.

The model uses transition distances and no_overlap constraints to model machines
setup times.
The no_overlap constraint is specified to enforce transition distance between
immediate successors on the sequence.
Forbidden transitions are modeled with a very large transition distance.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Number of types
NB_TYPES = 5

# Setup times of machine M1. -1 means forbidden transition
SETUP_M1 = [[0, 26,   8,  3, -1],
            [22,  0, -1,  4, 22],
            [28,  0,  0, 23,  9],
            [29, -1, -1,  0,  8],
            [26, 17, 11,  7,  0]]

# Setup times of machine M2. -1 means forbidden transition
SETUP_M2 = [[0,   5, 28, -1,  2],
            [-1,  0, -1,  7, 10],
            [19, 22,  0, 28, 17],
            [7,  26, 13,  0, -1],
            [13, 17, 26, 20,  0]]

# Task types
TASK_TYPE = [3, 3, 1, 1, 1, 1, 2, 0, 0, 2,
             4, 4, 3, 3, 2, 3, 1, 4, 4, 2,
             2, 1, 4, 2, 2, 0, 3, 3, 2, 1,
             2, 1, 4, 3, 3, 0, 2, 0, 0, 3,
             2, 0, 3, 2, 2, 4, 1, 2, 4, 3]

# Number of tasks
NB_TASKS = len(TASK_TYPE)

# Task duration if executed on machine M1
TASK_DUR_M1 = [ 4, 17,  4,  7, 17, 14,  2, 14, 2,  8,
               11, 14,  4, 18,  3,  2,  9,  2, 9, 17,
               18, 19,  5,  8, 19, 12, 17, 11, 6,  3,
               13,  6, 19,  7,  1,  3, 13,  5, 3,  6,
               11, 16, 12, 14, 12, 17,  8,  8, 6,  6]

# Task duration if executed on machine M2
TASK_DUR_M2 = [12,  3, 12, 15,  4,  9, 14,  2,  5,  9,
               10, 14,  7,  1, 11,  3, 15, 19,  8,  2,
               18, 17, 19, 18, 15, 14,  6,  6,  1,  2,
                3, 19, 18,  2,  7, 16,  1, 18, 10, 14,
                2,  3, 14,  1,  1,  6, 19,  5, 17,  4]


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Put max value for forbidden transitions in setup times
for i in range(NB_TYPES):
    for j in range(NB_TYPES):
        if SETUP_M1[i][j] < 0:
            SETUP_M1[i][j] = INTERVAL_MAX;
        if SETUP_M2[i][j] < 0:
            SETUP_M2[i][j] = INTERVAL_MAX;


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Build tasks for machine M1 and M2
tasks_m1 = [interval_var(name='A{}_M1_TP{}'.format(i, TASK_TYPE[i]), optional=True, size=TASK_DUR_M1[i]) for i in range(NB_TASKS)]
tasks_m2 = [interval_var(name='A{}_M2_TP{}'.format(i, TASK_TYPE[i]), optional=True, size=TASK_DUR_M1[i]) for i in range(NB_TASKS)]

# Build actual tasks as an alternative between machines
tasks = [interval_var(name='A{}_TP{}'.format(i, TASK_TYPE[i])) for i in range(NB_TASKS)]
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

# Minimize the makespan
mdl.add(minimize(max([end_of(tasks[i]) for i in range(NB_TASKS)])))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def compact(name):
    # Example: A31_M1_TP1 -> 31
    task, foo = name.split('_', 1)
    return task[1:]

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

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=100000, TimeLimit=10)
print('Solution: ')
res.print_solution()

if res and visu.is_visu_enabled():
    visu.timeline('Solution for SchedSetup')
    showsequence(s1, SETUP_M1)
    showsequence(s2, SETUP_M2)
    visu.show()
