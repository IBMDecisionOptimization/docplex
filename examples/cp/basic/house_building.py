# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This problem schedule a series of tasks of varying durations where some tasks must finish
before others start. And assign workers to each of the tasks such that each worker is assigned
to only one task to a given time. The objective of the problem is to maximize the matching worker
skill level to the tasks.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from collections import namedtuple


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Number of Houses to build
NB_HOUSES = 5

# Max number of periods for the schedule
MAX_SCHEDULE = 318
MAX_SCHEDULE = 200000

# House construction tasks
Task = (namedtuple("Task", ["name", "duration"]))
TASKS = {Task("masonry",   35),
         Task("carpentry", 15),
         Task("plumbing",  40),
         Task("ceiling",   15),
         Task("roofing",    5),
         Task("painting",  10),
         Task("windows",    5),
         Task("facade",    10),
         Task("garden",     5),
         Task("moving",     5),
        }

# The tasks precedences
TaskPrecedence = (namedtuple("TaskPrecedence", ["beforeTask", "afterTask"]))
TASK_PRECEDENCES = {TaskPrecedence("masonry",   "carpentry"),
                    TaskPrecedence("masonry",   "plumbing"),
                    TaskPrecedence("masonry",   "ceiling"),
                    TaskPrecedence("carpentry", "roofing"),
                    TaskPrecedence("ceiling",   "painting"),
                    TaskPrecedence("roofing",   "windows"),
                    TaskPrecedence("roofing",   "facade"),
                    TaskPrecedence("plumbing",  "facade"),
                    TaskPrecedence("roofing",   "garden"),
                    TaskPrecedence("plumbing",  "garden"),
                    TaskPrecedence("windows",   "moving"),
                    TaskPrecedence("facade",    "moving"),
                    TaskPrecedence("garden",    "moving"),
                    TaskPrecedence("painting",  "moving"),
                   }


# Workers Name and level for each of there skill
Skill = (namedtuple("Skill", ["worker", "task", "level"]))
SKILLS = {Skill("Joe",  "masonry",   9),
          Skill("Joe",  "carpentry", 7),
          Skill("Joe",  "ceiling",   5),
          Skill("Joe",  "roofing",   6),
          Skill("Joe",  "windows",   8),
          Skill("Joe",  "facade",    5),
          Skill("Joe",  "garden",    5),
          Skill("Joe",  "moving",    6),
          Skill("Jack", "masonry",   5),
          Skill("Jack", "plumbing",  7),
          Skill("Jack", "ceiling",   8),
          Skill("Jack", "roofing",   7),
          Skill("Jack", "painting",  9),
          Skill("Jack", "facade",    5),
          Skill("Jack", "garden",    5),
          Skill("Jim",  "carpentry", 5),
          Skill("Jim",  "painting",  6),
          Skill("Jim",  "windows",   5),
          Skill("Jim",  "garden",    9),
          Skill("Jim",  "moving",    8)
          }

# Worker and continuity requirements: if the Task 1 is done on the house, he must do the task 2 in this house
Continuity = (namedtuple("Continuity", ["worker", "task1", "task2"]))
CONTINUITIES = {Continuity("Joe",  "masonry",   "carpentry"),
                Continuity("Jack", "roofing",   "facade"),
                Continuity("Joe",  "carpentry", "roofing"),
                Continuity("Jim",  "garden",    "moving")
               }


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Find_tasks: return the task it refers to in the Tasks vector
def find_tasks(name):
    return next(t for t in TASKS if t.name == name)

# Find_skills: return the skill it refers to in the Skills vector
def find_skills(worker, task):
    return next(s for s in SKILLS if (s.worker == worker) and (s.task == task))

# Iterator on houses numbers
HOUSES = range(1, NB_HOUSES + 1)

# Build the list of all worker names
WORKERS = set(sk.worker for sk in SKILLS)


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Variables of the model
tasks = {}   # dict of interval variable for each house and task
wtasks = {}  # dict of interval variable for each house and skill
for house in HOUSES:
    for task in TASKS:
        v = (0, MAX_SCHEDULE)
        tasks[(house, task)] = mdl.interval_var(v, v, size=task.duration, name="house {} task {}".format(house, task))
    for task in SKILLS:
        wtasks[(house, task)] = mdl.interval_var(optional=True, name="house {} skill {}".format(house, task))

# Maximization objective of the model
obj2 = mdl.sum([s.level * mdl.presence_of(wtasks[(h, s)]) for s in SKILLS for h in HOUSES])
mdl.add(mdl.maximize(obj2))

# Constraints of the model
for h in HOUSES:
    # Temporal constraints
    for p in TASK_PRECEDENCES:
        mdl.add(mdl.end_before_start(tasks[(h, find_tasks(p.beforeTask))], tasks[(h, find_tasks(p.afterTask))]))
    # Alternative workers
    for t in TASKS:
        mdl.add(mdl.alternative(tasks[(h, t)], [wtasks[(h, s)] for s in SKILLS if (s.task == t.name)], 1))
    # Continuity constraints
    for c in CONTINUITIES:
        mdl.add(mdl.presence_of(wtasks[(h, find_skills(c.worker, c.task1))]) ==
                mdl.presence_of(wtasks[(h, find_skills(c.worker, c.task2))]))

# No overlap constraint
for w in WORKERS:
    mdl.add(mdl.no_overlap([wtasks[(h, s)] for h in HOUSES for s in SKILLS if s.worker == w]))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

print("\nSolving model....")
msol = mdl.solve(TimeLimit=20, trace_log=False)

# Print solution
print("Solve status: " + msol.get_solve_status())
if msol.is_solution():
    # Sort tasks in increasing begin order
    ltasks = []
    for hs in HOUSES:
        for tsk in TASKS:
            (beg, end, dur) = msol[tasks[(hs, tsk)]]
            ltasks.append((hs, tsk, beg, end, dur))
    ltasks = sorted(ltasks, key = lambda x : x[2])
    # Print solution
    print("\nList of tasks in increasing start order:")
    for tsk in ltasks:
        print("From " + str(tsk[2]) + " to " + str(tsk[3]) + ", " + tsk[1].name + " in house " + str(tsk[0]))

