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

This model has been enriched by the addition of KPIs (key performance indicators), operational with a
version of COS greater or equal to 12.9.0.0.
These are named expressions which are of interest to help get an idea of the performance of the model.
Here, we are interested in two indicators:
 - the first is the `occupancy'' defined as the total demand divided by the total plant capacity.
 - the second indicator is the occupancy which is the lowest of all the plants.

The KPIs are displayed using a SolverProgressPanelListener that displays solve progress in real time
and allows to stop solve when good enough objective or KPIs are reached.
Log parsing is also activated to retrieve runtime information from it.
"""

from docplex.cp.model import *
from docplex.cp.solver.solver_listener import *
from docplex.cp.config import context
from docplex.cp.utils import compare_natural
from collections import deque
import os

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Read problem data from a file and convert it as a list of integers
filename = os.path.dirname(os.path.abspath(__file__)) + '/data/plant_location.data'
data = deque()
with open(filename, 'r') as file:
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
cust = integer_var_list(nbCustomer, 0, nbLocation - 1, 'CustomerLocation')

# Create variables indicating which plant location is open
open = integer_var_list(nbLocation, 0, 1, 'OpenLocation')

# Create variables indicating load of each plant
load = [integer_var(0, capacity[p], 'PlantLoad_' + str(p)) for p in range(nbLocation)]

# Associate plant openness to its load
mdl.add(open[p] == (load[p] > 0) for p in range(nbLocation))

# Add constraints
mdl.add(pack(load, cust, demand))

# Add objective
obj = scal_prod(fixedCost, open) + sum(element(cust[c], cost[c]) for c in range(nbCustomer))
mdl.add(minimize(obj))

# Add KPIs
if compare_natural(context.model.version, '12.9') >= 0:
    mdl.add_kpi(sum(demand) / scal_prod(open, capacity), 'Average Occupancy')
    mdl.add_kpi(min([load[l] / capacity[l] + (1 - open[l]) for l in range(nbLocation)]), 'Min occupancy')


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

if context.visu_enabled:
    mdl.add_solver_listener(SolverProgressPanelListener(parse_log=True))

# Solve the model
print('Solve the model')
res = mdl.solve(TimeLimit=20, LogPeriod=1000)
res.write()
