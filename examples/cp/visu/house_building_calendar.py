# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This is a problem of building five houses. The masonry, roofing,
painting, etc. must be scheduled. Some tasks must necessarily take
place before others and these requirements are expressed through
precedence constraints.

There are two workers and each task requires a specific worker.
The worker has a calendar of days off that must be taken into account.
The objective is to minimize the overall completion date.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# List of available workers together with their holidays as list of tuples (start_day, end_day)
WORKERS = {
 'Joe' :  [ (5, 12), (124, 131), (215, 236), (369, 376), (495, 502), (579, 600) ],
 'Jim' :  [ (26, 40), (201, 225), (306, 313), (397, 411), (565, 579) ]
}

# List of tasks to be executed for each house
TASKS = {
  'masonry'   : (35 , 'Joe',  1),
  'carpentry' : (15 , 'Joe',  2),
  'plumbing'  : (40 , 'Jim',  3),
  'ceiling'   : (15 , 'Jim',  4),
  'roofing'   : ( 5 , 'Joe',  5),
  'painting'  : (10 , 'Jim',  6),
  'windows'   : ( 5 , 'Jim',  7),
  'facade'    : (10 , 'Joe',  8),
  'garden'    : ( 5 , 'Joe',  9),
  'moving'    : ( 5 , 'Jim', 10)
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

# Total number of houses to build
NUMBER_OF_HOUSES = 5

# Max number of calendar years
MAX_YEARS = 2

#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Initialize availability calendar for workers

calendars = { w : CpoStepFunction() for w in WORKERS }
for w in WORKERS:
    calendars[w].set_value(0, MAX_YEARS * 365, 100)
    # Remove week ends
    for i in range(MAX_YEARS * 52):
        calendars[w].set_value(5 + (7 * i), 7 + (7 * i), 0)
    # Remove holidays
    for s,e in WORKERS[w]:
        calendars[w].set_value(s, e, 0)

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Initialize model variable sets
worker_tasks = { w : [] for w in WORKERS}  # Tasks assigned to workers (key is the worker)
house_finish_times = []  # Array of house finishing times

# Utility function
def make_house(loc):
    ''' Create model elements corresponding to the building of one house
    loc: Identification (index) of the house to build
    '''

    # Create interval variable for each task for this house
    tasks = { t: interval_var(size=TASKS[t][0], intensity=calendars[TASKS[t][1]], name='H{}-{}'.format(loc,t)) for t in TASKS }
    for t in TASKS:
        mdl.forbid_start(tasks[t], calendars[TASKS[t][1]])
        mdl.forbid_end  (tasks[t], calendars[TASKS[t][1]])

    # Add precedence constraints
    mdl.add(end_before_start(tasks[p], tasks[s]) for p,s in PRECEDENCES)

    # Allocate tasks to workers
    for t in TASKS:
        worker_tasks[TASKS[t][1]].append(tasks[t])

    # Update cost
    house_finish_times.append(end_of(tasks['moving']))


# Make houses
for i in range(NUMBER_OF_HOUSES):
    make_house(i)

# Avoid each worker tasks overlapping
mdl.add(no_overlap(worker_tasks[w]) for w in WORKERS)

# Add minimization objective
mdl.add(minimize(max(house_finish_times)))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def compact(name):
    # Example: H3-garden -> G3
    #           ^ ^
    loc, task = name[1:].split('-', 1)
    # Returns color index and compacted name
    return int(TASKS[task][2]), (task[0].upper() + loc)

# Solve model
print('Solving model...')
res = mdl.solve(TimeLimit=10)
print('Solution:')
res.print_solution()

# Display result
import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    visu.timeline('Solution house building with calendars')
    visu.panel()
    for w in WORKERS:
        visu.pause(calendars[w])
        visu.sequence(name=w)
        for t in worker_tasks[w]:
            visu.interval(res.get_var_solution(t), *compact(t.get_name()))
    visu.show()
