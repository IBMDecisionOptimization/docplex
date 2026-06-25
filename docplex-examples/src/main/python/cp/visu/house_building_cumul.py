# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This is a problem of building five houses in different locations.
The masonry, roofing, painting, etc. must be scheduled.
Some tasks must necessarily take place before others and these requirements are
expressed through precedence constraints.

There are three workers, and each task requires a worker.
There is also a cash budget which starts with a given balance.
Each task costs a given amount of cash per day which must be available at the start of the task.
A cash payment is received periodically.
The objective is to minimize the overall completion date.

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

# Number of workers
NB_WORKERS = 3

# List of houses to build. Value is the minimum start date
HOUSES = [ 31, 0, 90, 120, 90 ]

# Cash parameters
NB_PAYMENTS = 5
PAYMENT_AMOUNT = 30000
PAYMENT_INTERVAL = 60

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Initialize cumul function
workers_usage = 0  # Total worker usage
cash = sum(step_at(PAYMENT_INTERVAL * p, PAYMENT_AMOUNT) for p in range(NB_PAYMENTS))

all_tasks = []  # Array of all tasks

# Utility function
def make_house(loc, rd):
    ''' Create model elements corresponding to the building of one house
    loc: Identification (index) of the house to build
    rd:  Minimal start date
    '''

    # Create interval variable for each task for this house
    tasks = { t: interval_var(size=TASKS[t], name='H{}-{}'.format(loc,t)) for t in TASKS }

    # Add precedence constraints
    mdl.add(end_before_start(tasks[p], tasks[s]) for p,s in PRECEDENCES)

    # Update cumul functions
    global workers_usage
    global cash
    workers_usage += sum(pulse(tasks[t], 1) for t in TASKS)
    cash          -= sum(step_at_start(tasks[t], 200 * TASKS[t]) for t in TASKS)

    # Update cost
    all_tasks.extend(tasks.values())


# Make houses
for i, sd in enumerate(HOUSES):
    make_house(i, sd)

# Number of workers should not be greater than the limit
mdl.add(workers_usage <= NB_WORKERS)

# Cash should not be negative
mdl.add(cash >= 0)

# Minimize overall completion date
mdl.add(minimize(max(end_of(task) for task in all_tasks)))


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

# Display result
import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    workersF = CpoStepFunction()
    cashF = CpoStepFunction()
    for p in range(NB_PAYMENTS):
        cashF.add_value(PAYMENT_INTERVAL * p, INT_MAX, PAYMENT_AMOUNT)
    for task in all_tasks:
        itv = res.get_var_solution(task)
        workersF.add_value(itv.get_start(), itv.get_end(), 1)
        cashF.add_value(itv.start, INT_MAX, -200 * itv.get_length())
    visu.timeline('Solution SchedCumul')
    visu.panel(name='Schedule')
    for task in all_tasks:
        visu.interval(res.get_var_solution(task), *compact(task.get_name()))
    visu.panel(name='Workers')
    visu.function(segments=workersF, style='area')
    visu.panel(name='Cash')
    visu.function(segments=cashF, style='area', color='gold')
    visu.show()
