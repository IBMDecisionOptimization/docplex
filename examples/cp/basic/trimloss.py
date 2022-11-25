# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2020
# --------------------------------------------------------------------------

"""
The trim loss problems arises in the paper industry.

The problem is to cut wide papers rolls into sub rolls (orders).
The wide roll are cut into pieces with a cutting pattern.

A cutting pattern defines the blades positions for cutting the roll.
A maximum number of orders is allowed in a cutting pattern (here it is 6).
When cutting a wide roll, we can have a loss of paper that is wasted.
This loss is contrained to be not more than a given value (here it is 100)

An order is characterised by a demand, a roll width, and a maximum number of
time it can appear in a cutting pattern.

The goal is to meet the demand while minimizing the roll used and the number
of different cutting patterns used for production.

In this example we also use:
- extra constraints to avoid assigning orders to unused patterns,
- lexicographic constraints to break symmetries between cutting patterns
- strong constraints to have a better domain reduction by enumerating possible
  patterns configurations
All this makes the proof of optimality rather fast.
"""

from docplex.cp.model import *
from sys import stdout

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Data
ROLL_WIDTH    = 2200    # Width of roll to be cutted into pieces
MAX_WASTE     = 100     # Maximum waste per roll
MAX_ORDER_PER_CUT = 5   # Maximum number of order per cutting pattern 

# Orders demand, width and max occurence in a cutting pattern
ORDER_DEMAND     = (  8,  16,  12,   7,  14,  16)
ORDER_WIDTH      = (330, 360, 380, 430, 490, 530)
ORDER_MAX_REPEAT = (  2,   3,   3,   5,   3,   4)
# Number of different order types 
NUM_ORDER_TYPE = len(ORDER_DEMAND)
# Maximum number of cutting pattern
NUM_PATTERN_TYPE = 6
# Maximum of time a cutting pattern is used
MAX_PATTERN_USAGE = 16
# Cost of using a pattern
PATTERN_COST  = 0.1
# Cost of a roll
ROLL_COST = 1

PATTERNS = range(NUM_PATTERN_TYPE)
ORDERS   = range(NUM_ORDER_TYPE)


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

model = CpoModel()

# Decision variables : pattern usage
patternUsage = [model.integer_var(0, MAX_PATTERN_USAGE, "PatternUsage_"+str(p)) for p in PATTERNS]

# Decision variables : order quantity per pattern
x = [[model.integer_var(0, max, "x["+str(o)+","+str(p)+"]")
        for (o, max) in enumerate(ORDER_MAX_REPEAT)]
        for p in PATTERNS]

# Maximum number of orders per cutting pattern
for p in PATTERNS :
    model.add(sum(x[p]) <= MAX_ORDER_PER_CUT)

# Roll capacity
usage = [0] + [v for v in range(ROLL_WIDTH - MAX_WASTE, ROLL_WIDTH+1)]   # usage is [0, 2100..2200]
rollUsage = [model.integer_var(domain = usage, name = "RollUsage_"+str(p)) for p in PATTERNS]

for p in PATTERNS :
    model.add(sum(ORDER_WIDTH[o] * x[p][o] for o in ORDERS) == rollUsage[p])

# Production requirement
for o in ORDERS :
    model.add(model.sum(x[p][o] * patternUsage[p] for p in PATTERNS) >= ORDER_DEMAND[o])

# Objective
model.add(minimize(model.sum((patternUsage[p] > 0) * PATTERN_COST + patternUsage[p] * ROLL_COST
                             for p in PATTERNS)))

# Extra constraint to avoid assigning orders to an unused pattern
for p in PATTERNS :
    model.add((patternUsage[p] == 0) == (rollUsage[p] == 0))

# Extra lexicographic constraint to break symmetries
for p in range(NUM_PATTERN_TYPE - 1) :
    model.add(model.lexicographic([patternUsage[p]] + x[p], [patternUsage[p+1]] + x[p+1]))

# Strong constraints to improve the time to prove optimality
for p in PATTERNS :
    model.add(model.strong(x[p]))

# KPIs : Number of rolls, of pattern used and total loss of paper
model.add_kpi(model.sum([patternUsage[p] for p in PATTERNS]), "Rolls")
model.add_kpi(model.sum([(patternUsage[p] > 0) for p in PATTERNS]), "Patterns")
model.add_kpi(model.sum([patternUsage[p] * (ROLL_WIDTH - rollUsage[p]) for p in PATTERNS]), "Loss")


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

print("Solve the model...")
msol = model.solve(LogPeriod=1000000, TimeLimit=300)
if msol:
    print("patternUsage = ")
    for p in PATTERNS:
        l = ROLL_WIDTH - msol[rollUsage[p]]
        stdout.write("Pattern {} , usage = {}, roll usage = {}, loss = {}, orders =".format(p, msol[patternUsage[p]], msol[rollUsage[p]], l))
        for o in ORDERS:
            stdout.write(" {}".format(msol[x[p][o]]))
        stdout.write('\n')
else:
    print("No solution found")
