# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The problem is to build steel coils from slabs that are available in a
work-in-process inventory of semi-finished products.
There is no limitation in the number of slabs that can be requested,
but only a finite number of slab sizes is available
(sizes 11, 13, 16, 17, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 40, 43, 45).
The problem is to select a number of slabs to build the coil orders,
and to satisfy the following constraints:

    * A coil order can be built from only one slab.
    * Each coil order requires a specific process to build it from a
      slab. This process is encoded by a "color".
    * Several coil orders can be built from the same slab, but a slab can
      be used to produce at most two different "colors" of coils.
    * The sum of the sizes of each coil order built from a slab must not
      exceed the slab size.

Finally, the production plan should minimize the unused capacity of the
selected slabs.

This problem is based on "prob038: Steel mill slab design problem" from
CSPLib (www.csplib.org). It is a simplification of an industrial problem
described in J. R. Kalagnanam, M. W. Dawande, M. Trumbo, H. S. Lee.
"Inventory Matching Problems in the Steel Industry," IBM Research
Report RC 21171, 1998.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from collections import namedtuple


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# List of coils to produce (orders)
Order = namedtuple("Order", ['id', 'weight', 'color'])
ORDERS = (
            Order( 1, 22, 5),
            Order( 2,  9, 3),
            Order( 3,  9, 4),
            Order( 4,  8, 5),
            Order( 5,  8, 7),
            Order( 6,  6, 3),
            Order( 7,  5, 6),
            Order( 8,  3, 0),
            Order( 9,  3, 2),
            Order(10,  3, 3),
            Order(11,  2, 1),
            Order(12,  2, 5)
         )

# Max number of different colors of coils produced by a single slab
MAX_COLOR_PER_SLAB = 2

# List of available slab weights.
AVAILABLE_SLAB_WEIGHTS = [11, 13, 16, 17, 19, 20, 23, 24, 25,
                          26, 27, 28, 29, 30, 33, 34, 40, 43, 45]


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Upper bound for the number of slabs to use
MAX_SLABS = len(ORDERS)

# Build a set of all colors
allcolors = set(o.color for o in ORDERS)

# The heaviest slab
max_slab_weight = max(AVAILABLE_SLAB_WEIGHTS)

# Minimum loss incurred for a given slab usage.
# loss[v] = loss when smallest slab is used to produce a total weight of v
loss = [0] + [min([sw - use for sw in AVAILABLE_SLAB_WEIGHTS if sw >= use]) for use in range(1, max_slab_weight + 1)]


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model 
mdl = CpoModel()

# Index of the slab used to produce each coil order
production_slab = mdl.integer_var_list(len(ORDERS), 0, MAX_SLABS - 1, "production_slab")

# Usage of each slab
slab_use = mdl.integer_var_list(MAX_SLABS, 0, max_slab_weight, "slab_use")

# The orders are allocated to the slabs with capacity
mdl.add(mdl.pack(slab_use, production_slab, [o.weight for o in ORDERS]))

# Constrain max number of colors produced by each slab
for s in range(MAX_SLABS):
   su = 0
   for c in allcolors:
       lo = False
       for i, o in enumerate(ORDERS):
           if o.color == c:
               lo |= (production_slab[i] == s)
       su += lo
   mdl.add(su <= MAX_COLOR_PER_SLAB)

# Minimize the total loss
total_loss = sum([mdl.element(slab_use[s], loss) for s in range(MAX_SLABS)])
mdl.add(mdl.minimize(total_loss))

# Set search strategy
mdl.set_search_phases([mdl.search_phase(production_slab)])


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print("Solving model....")
msol = mdl.solve(FailLimit=100000, TimeLimit=10)

# Print solution
if msol:
    print("Solution: ")
    for s in set(msol[ps] for ps in production_slab):
        # Determine orders using this slab
        lordrs = [o for i, o in enumerate(ORDERS) if msol[production_slab[i]] == s]
        # Compute display attributes
        used_weight = msol[slab_use[s]]          # Weight used in the slab
        loss_weight = loss[used_weight]          # Loss weight
        colors = set(o.color for o in lordrs)    # List of colors
        loids = [o.id for o in lordrs]           # List of order irs
        print("Slab weight={}, used={}, loss={}, colors={}, orders={}"
              .format(used_weight + loss_weight, used_weight, loss_weight, colors, loids))
else:
    print("No solution found")




    
    



