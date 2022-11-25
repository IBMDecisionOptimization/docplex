# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
A ship-building company has a certain number of customers. Each customer is supplied
by exactly one plant. In turn, a plant can supply several customers. The problem is
to decide where to set up the plants in order to supply every customer while minimizing
the cost of building each plant and the transportation cost of supplying the customers.

For each possible plant location there is a fixed cost and a production capacity.
Both take into account the country and the geographical conditions.

For every customer, there is a demand and a transportation cost with respect to
each plant location.

While a first solution of this problem can be found easily by CP Optimizer, it can take
quite some time to improve it to a very good one. We illustrate the warm start capabilities
of CP Optimizer by giving a good starting point solution that CP Optimizer will try to improve.
This solution could be one from an expert or the result of another optimization engine
applied to the problem.

In the solution we only give a value to the variables that determine which plant delivers
a customer. This is sufficient to define a complete solution on all model variables.
CP Optimizer first extends the solution to all variables and then starts to improve it.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from docplex.cp.solution import CpoModelSolution
from collections import deque
import os

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Read problem data from a file and convert it as a list of integers
filename = os.path.dirname(os.path.abspath(__file__)) + "/data/plant_location.data"
data = deque()
with open(filename, "r") as file:
    for val in file.read().split():
        data.append(int(val))

# Read number of customers and locations
nbCustomer = data.popleft()
nbLocation = data.popleft()

# Initialize cost. cost[c][p] = cost to deliver customer c from plant p
cost = list([list([data.popleft() for l in range(nbLocation)]) for c in range(nbCustomer)])

# Initialize demand of each customer
demand = list([data.popleft() for c in range(nbCustomer)])

# Initialize fixed cost of each location
fixedCost = list([data.popleft() for p in range(nbLocation)])

# Initialize capacity of each location
capacity = list([data.popleft() for p in range(nbLocation)])


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

mdl = CpoModel()

# Create variables identifying which location serves each customer
cust = mdl.integer_var_list(nbCustomer, 0, nbLocation - 1, "CustomerLocation")

# Create variables indicating which plant location is open
open = mdl.integer_var_list(nbLocation, 0, 1, "OpenLocation")

# Create variables indicating load of each plant
load = [mdl.integer_var(0, capacity[p], "PlantLoad_" + str(p)) for p in range(nbLocation)]

# Associate plant openness to its load
for p in range(nbLocation):
      mdl.add(open[p] == (load[p] > 0))

# Add constraints
mdl.add(mdl.pack(load, cust, demand))

# Add objective
obj = mdl.scal_prod(fixedCost, open)
for c in range(nbCustomer):
    obj += mdl.element(cust[c], cost[c])
mdl.add(mdl.minimize(obj))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve without starting point
print("Solve the model with no starting point")
msol = mdl.solve(TimeLimit=10)
if msol:
    print("   Objective value: " + str(msol.get_objective_values()[0]))
else:
    print("   No solution")

# Solve with starting point
print("Solve the model with starting point")
custValues = [19,  0, 11,  8, 29,  9, 29, 28, 17, 15,  7,  9, 18, 15,  1, 17, 25, 18, 17, 27,
              22,  1, 26,  3, 22,  2, 20, 27,  2, 16,  1, 16, 12, 28, 19,  2, 20, 14, 13, 27,
               3,  9, 18,  0, 13, 19, 27, 14, 12,  1, 15, 14, 17,  0,  7, 12, 11,  0, 25, 16,
              22, 13, 16,  8, 18, 27, 19, 23, 26, 13, 11, 11, 19, 22, 28, 26, 23,  3, 18, 23,
              26, 14, 29, 18,  9,  7, 12, 27,  8, 20]
sp = CpoModelSolution()
for c in range(nbCustomer):
    sp.add_integer_var_solution(cust[c], custValues[c])
mdl.set_starting_point(sp)

try:
    msol = mdl.solve(TimeLimit=10)
    if msol:
        print("   Objective value: " + str(msol.get_objective_values()[0]))
    else:
        print("   No solution")
except:
    print("   Starting point seems not available with your solver version.")
