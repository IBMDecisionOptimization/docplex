# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
In mathematics, a Golomb ruler is a set of marks at integer positions along
an imaginary ruler such that no two pairs of marks are the same distance apart.
The number of marks on the ruler is its order, and the largest distance
between two of its marks is its length.

This implementation differs from the 'basic' one because it calls the solver
twice:
 * First time to know the minimal size of the ruler for the required order,
 * A second time to list all possible rulers for this optimal size

See https://en.wikipedia.org/wiki/Golomb_ruler for more information.

For order 5: 2 solutions 0 1 4 9 11 ; 0 2 7 8 11   

Please refer to documentation for appropriate setup of solving configuration.
"""


from docplex.cp.model import CpoModel

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Number of marks on the ruler
ORDER = 5


#-----------------------------------------------------------------------------
# Prepare the data for modeling (Voluntarily wrong)
#-----------------------------------------------------------------------------

# Estimate an upper bound to the ruler length
MAX_LENGTH = 10  # TOO SHORT


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create array of variables corresponding to position rule marks
marks = mdl.integer_var_list(ORDER, 0, MAX_LENGTH, "M")

# Create marks distances that should be all different
dist = [marks[i] - marks[j] for i in range(1, ORDER) for j in range(0, i)]
mdl.add(mdl.all_diff(dist))

# Avoid symmetric solutions by ordering marks
mdl.add(marks[0] == 0)
for i in range(1, ORDER):
    mdl.add(marks[i] > marks[i - 1])

# Avoid mirror solution
mdl.add((marks[1] - marks[0]) < (marks[ORDER - 1] - marks[ORDER - 2]))

# Minimize ruler size (position of the last mark)
mdl.add(mdl.minimize(marks[ORDER - 1]))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# First solve the model
msol = mdl.solve()
# if msol:
#     msol.print_solution()
# else:
if True:
    rsol = mdl.refine_conflict()
    rsol.print_conflict()
