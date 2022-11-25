# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This is a problem of building five houses.
The masonry, roofing, painting, etc. must be scheduled.
Some tasks must necessarily take place before others and these requirements
are expressed through precedence constraints.

A pool of two workers is available for building the houses.
For a given house, some tasks (namely: plumbing, ceiling and painting) require
the house to be clean whereas other tasks (namely: masonry, carpentry, roofing
and windows) put the house in a dirty state.
A transition time of 1 is needed to clean the house so to change from state
'dirty' to state 'clean'.

The objective is to minimize the makespan.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# List of tasks to be executed for each house
TASKS = {
  'masonry'   : 35,
  'carpentry' : 15,
  'plumbing'  : 40,
  'ceiling'   : 15,
  'roofing'   :  5,
  'painting'  : 10,
  'windows'   :  5,
  'facade'    : 10,
  'garden'    :  5,
  'moving'    :  5
}

# Tasks precedence constraints (each tuple (X, Y) means X ends before start of Y)
PRECEDENCES = [
  ('masonry',   'carpentry'),
  ('masonry',   'plumbing'),
  ('masonry',   'ceiling'),
  ('carpentry', 'roofing'),
  ('ceiling',   'painting'),
  ('roofing',   'windows'),
  ('roofing',   'facade'),
  ('plumbing',  'facade'),
  ('roofing',   'garden'),
  ('plumbing',  'garden'),
  ('windows',   'moving'),
  ('facade',    'moving'),
  ('garden',    'moving'),
  ('painting',  'moving'),
]

# List of tasks that requires the house to be clean
CLEAN_TASKS = [ 'plumbing', 'ceiling', 'painting' ]

# List of tasks that put the house in a dirty state
DIRTY_TASKS = [ 'masonry', 'carpentry', 'roofing', 'windows' ]

# House cleaning transition time
HOUSE_CLEANING_TIME = 1

# Number of workers
NB_WORKERS = 2

# List of houses to build. Value is the minimum start date
HOUSES = (31, 0, 90, 120, 90)

# Possible states
CLEAN = 0
DIRTY = 1

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Initialize model variable sets
all_tasks = []  # Array of all tasks
workers_usage = 0  # Total worker usage
all_state_functions = []

# Transition matrix for cost between house states
TTIME = [[0, 0],[0, 0]]
TTIME[DIRTY][CLEAN] = HOUSE_CLEANING_TIME

# Utility function
def make_house(loc, rd):
    ''' Create model elements corresponding to the building of a house
        loc: Identification of house location
        rd:  Min start date
    '''

    # Create interval variable for each task for this house
    tasks = { t : interval_var(size=TASKS[t], start=(rd, INTERVAL_MAX), name='H{}-{}'.format(loc,t)) for t in TASKS }
    all_tasks.extend(tasks.values())

    global workers_usage

    # Add precedence constraints
    mdl.add(end_before_start(tasks[p], tasks[s]) for p,s in PRECEDENCES)

    # Create house state function
    house_state = state_function(TTIME, name='H{}'.format(loc))
    for t in CLEAN_TASKS:
        mdl.add(always_equal(house_state, tasks[t], CLEAN))
    for t in DIRTY_TASKS:
        mdl.add(always_equal(house_state, tasks[t], DIRTY))
    all_state_functions.append(house_state)

    # Allocate tasks to workers
    workers_usage += sum(pulse(tasks[t], 1) for t in TASKS)


# Make houses
for i, sd in enumerate(HOUSES):
    make_house(i, sd)

# Number of workers should not be greater than the limit
mdl.add(always_in(workers_usage, (INTERVAL_MIN, INTERVAL_MAX), 0, NB_WORKERS))

# Minimize overall completion date
mdl.add(minimize(max([end_of(task) for task in all_tasks])))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def compact(name):
    # Example: H3-garden -> G3
    #           ^ ^
    loc, task = name[1:].split('-', 1)
    return int(loc), task[0].upper() + loc

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=10000,TimeLimit=10)

print('Solution:')
res.print_solution()

# Draw solution
import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    workers_function = CpoStepFunction()
    for v in all_tasks:
        itv = res.get_var_solution(v)
        workers_function.add_value(itv.get_start(), itv.get_end(), 1)
    visu.timeline('Solution SchedState')
    visu.panel(name='Schedule')
    for v in all_tasks:
        visu.interval(res.get_var_solution(v), *compact(v.get_name()))
    visu.panel(name='Houses state')
    for f in all_state_functions:
        visu.sequence(name=f.get_name(), segments=res.get_var_solution(f))
    visu.panel(name='Nb of workers')
    visu.function(segments=workers_function, style='line')
    visu.show()
