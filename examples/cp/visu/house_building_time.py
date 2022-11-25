# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This is a problem of building a house. The masonry, roofing, painting,
etc. must be scheduled.  Some tasks must necessarily take place before
others and these requirements are expressed through precedence
constraints.

Moreover, there are earliness and tardiness costs associated with some tasks.
The objective is to minimize these costs.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# List of tasks to be executed for the house
TASKS = {
  'masonry'   : (35 ,  1, {'release_date':25, 'earliness_cost':200.0} ),
  'carpentry' : (15 ,  2, {'release_date':75, 'earliness_cost':300.0} ),
  'plumbing'  : (40 ,  3, {} ),
  'ceiling'   : (15 ,  4, {'release_date':75, 'earliness_cost':100.0} ),
  'roofing'   : ( 5 ,  5, {} ),
  'painting'  : (10 ,  6, {} ),
  'windows'   : ( 5 ,  7, {} ),
  'facade'    : (10 ,  8, {} ),
  'garden'    : ( 5 ,  9, {} ),
  'moving'    : ( 5 , 10, {'due_date':100, 'tardiness_cost':400.0} )
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

# Create interval variable for each building task
tasks = { t: interval_var(size=TASKS[t][0], name=t) for t in TASKS }

# Add precedence constraints
mdl.add(end_before_start(tasks[p], tasks[s]) for p,s in PRECEDENCES)

# Cost function
fearliness = dict()   # Task earliness cost function
ftardiness = dict()   # Task tardiness cost function

for t in TASKS:
    if 'release_date' in TASKS[t][2]:
        fearliness[t] = CpoSegmentedFunction((-TASKS[t][2]['earliness_cost'], 0), [(TASKS[t][2]['release_date'], 0, 0)])
    if 'due_date' in TASKS[t][2]:
        ftardiness[t] = CpoSegmentedFunction((0, 0), [(TASKS[t][2]['due_date'], 0, TASKS[t][2]['tardiness_cost'],)])

# Minimize cost
mdl.add(minimize( sum( start_eval(tasks[t], fearliness[t]) for t in fearliness) +
                  sum( end_eval  (tasks[t], ftardiness[t]) for t in ftardiness) ))

#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

print('Solving model...')
res = mdl.solve(TimeLimit=10)
print('Solution:')
res.print_solution()

import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    visu.timeline('Solution house building', origin=10, horizon=120)
    visu.panel('Schedule')
    for t in TASKS:
        visu.interval(res.get_var_solution(tasks[t]), TASKS[t][1], t)
    for t in TASKS:
        itvsol = res.get_var_solution(tasks[t])
        if 'release_date' in TASKS[t][2]:
            visu.panel('Earliness')
            cost = fearliness[t].get_value(itvsol.get_start())
            visu.function(segments=[(itvsol, cost, t)], color=TASKS[t][1], style='interval')
            visu.function(segments=fearliness[t], color=TASKS[t][1])
        if 'due_date' in TASKS[t][2]:
            visu.panel('Tardiness')
            cost = ftardiness[t].get_value(itvsol.get_end())
            visu.function(segments=[(itvsol, cost, t)], color=TASKS[t][1], style='interval')
            visu.function(segments=ftardiness[t], color=TASKS[t][1])
    visu.show()
