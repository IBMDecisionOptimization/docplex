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

There are three workers, and each worker has a given non-negative
skill level for each task.  Each task requires one worker that will
have to be selected among the ones who have a non null skill level for
that task.  A worker can be assigned to only one task at a time.  Each
house has a deadline. The objective is to maximize the skill levels of
the workers assigned to the tasks while respecting the deadlines.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

NB_HOUSES = 5
DEADLINE = 318
WORKERS = ['Joe', 'Jack', 'Jim']
NB_WORKERS = len(WORKERS)

# List of tasks to be executed for each house
TASKS = {
  'masonry'   : (35, [9, 5, 0],  1),
  'carpentry' : (15, [7, 0, 5],  2),
  'plumbing'  : (40, [0, 7, 0],  3),
  'ceiling'   : (15, [5, 8, 0],  4),
  'roofing'   : ( 5, [6, 7, 0],  5),
  'painting'  : (10, [0, 9, 6],  6),
  'windows'   : ( 5, [8, 0, 5],  7),
  'facade'    : (10, [5, 5, 0],  8),
  'garden'    : ( 5, [5, 5, 9],  9),
  'moving'    : ( 5, [6, 0, 8], 10)
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

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Initialize model variable sets
total_skill = 0                                 # Expression computing total of skills
worker_tasks = [[] for w in range(NB_WORKERS)]  # Tasks (interval variables) assigned to a each worker

# Utility function
def make_house(loc, deadline):
    ''' Create model elements corresponding to the building of a house
    loc      Identification of house location
    deadline Deadline for finishing the house
    '''

    # Create interval variable for each task for this house
    tasks = {t: interval_var(size=TASKS[t][0], end=(0, deadline), name='H{}-{}'.format(loc,t)) for t in TASKS}

    # Add precedence constraints
    mdl.add(end_before_start(tasks[p], tasks[s]) for p,s in PRECEDENCES)

    # Allocate tasks to workers
    global total_skill
    allocs = { (t,w) : interval_var(optional=True, name='H{}-{}-{}'.format(loc, t, w)) for t in TASKS for w in range(NB_WORKERS) if TASKS[t][1][w] > 0 }
    total_skill += sum((TASKS[t][1][w] * presence_of(allocs[t,w])) for t,w in allocs)
    for t in TASKS:
        mdl.add(alternative(tasks[t], [allocs[t2,w] for t2,w in allocs if t==t2]))
    for t,w in allocs:
        worker_tasks[w].append(allocs[t,w])

# Make houses
for h in range(NB_HOUSES):
    make_house(h, DEADLINE)

# Avoid overlapping between tasks of each worker
for w in range(NB_WORKERS):
    mdl.add(no_overlap(worker_tasks[w]))

# Maximize total of skills
mdl.add(maximize(total_skill))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def compact(name):
    # Example: H3-garden -> G3
    #           ^ ^
    loc, task, worker = name[1:].split('-', 2)
    # Returns color index and compacted name
    return int(TASKS[task][2]), task[0].upper() + loc

# Solve model
print('Solving model...')
res = mdl.solve(FailLimit=10000, TimeLimit=10)

print('Solution:')
res.print_solution()

# Draw solution
import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    visu.timeline('Solution house building', 0, DEADLINE)
    for w in range(NB_WORKERS):
        visu.sequence(name=WORKERS[w])
        for t in worker_tasks[w]:
            wt = res.get_var_solution(t)
            if wt.is_present():
                visu.interval(wt, *compact(wt.get_name()))
    visu.show()
