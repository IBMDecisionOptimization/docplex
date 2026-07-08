# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
A company has 8 stores.
Each store must be supplied by one warehouse.
The company has 5 possible locations where it has property and can build a
supplier warehouse: Bonn, Bordeaux, London, Paris, and Rome.

The warehouse locations have different capacities. A warehouse built in Bordeaux
or Rome could supply only one store ; a warehouse built in London could supply
two stores; a warehouse built in Bonn could supply three stores; and a warehouse
built in Paris could supply four stores.

The supply costs vary for each store, depending on which warehouse is the
supplier. For example, a store that is located in Paris would have low supply
costs if it were supplied by a warehouse also in Paris.  That same store would
have much higher supply costs if it were supplied by the other warehouses.

The cost of building a warehouse varies depending on warehouse location.

The problem is to find the most cost-effective solution to this problem, while
making sure that each store is supplied by a warehouse.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from collections import namedtuple

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

Warehouse = namedtuple('Warehouse', ('city',      # Name of the city
                                     'capacity',  # Capacity of the warehouse
                                     'cost',      # Warehouse building cost
                                     ))

# List of warehouses
WAREHOUSES = (Warehouse("Bonn",     3, 480),
              Warehouse("Bordeaux", 1, 200),
              Warehouse("London",   2, 320),
              Warehouse("Paris",    4, 340),
              Warehouse("Rome",     1, 300))
NB_WAREHOUSES = len(WAREHOUSES)

# Number of stores
NB_STORES = 8

# Supply cost for each store and warehouse
SUPPLY_COST = ((24, 74, 31, 51, 84),
               (57, 54, 86, 61, 68),
               (57, 67, 29, 91, 71),
               (54, 54, 65, 82, 94),
               (98, 81, 16, 61, 27),
               (13, 92, 34, 94, 87),
               (54, 72, 41, 12, 78),
               (54, 64, 65, 89, 89))


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create CPO model
mdl = CpoModel()

# Create one variable per store to contain the index of its supplying warehouse
NB_WAREHOUSES = len(WAREHOUSES)
supplier = mdl.integer_var_list(NB_STORES, 0, NB_WAREHOUSES - 1, "supplier")

# Create one variable per warehouse to indicate if it is open (1) or not (0)
open = mdl.integer_var_list(NB_WAREHOUSES, 0, 1, "open")

# Add constraints stating that the supplying warehouse of each store must be open
for s in supplier:
    mdl.add(mdl.element(open, s) == 1)

# Add constraints stating that the number of stores supplied by each warehouse must not exceed its capacity
for wx in range(NB_WAREHOUSES):
    mdl.add(mdl.count(supplier, wx) <= WAREHOUSES[wx].capacity)

# Build an expression that computes total cost
total_cost = mdl.scal_prod(open, [w.cost for w in WAREHOUSES])
for sx in range(NB_STORES):
    total_cost = total_cost + mdl.element(supplier[sx], SUPPLY_COST[sx])

# Minimize total cost
mdl.add(mdl.minimize(total_cost))
 

#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=10)

# Print solution
if msol:
    for wx in range(NB_WAREHOUSES):
        if msol[open[wx]] == 1:
            print("Warehouse '{}' open to supply stores: {}"
                  .format(WAREHOUSES[wx].city,
                          ", ".join(str(sx) for sx in range(NB_STORES) if msol[supplier[sx]] == wx)))
    print("Total cost is: {}".format(msol.get_objective_values()[0]))
else:
    print("No solution found.")
