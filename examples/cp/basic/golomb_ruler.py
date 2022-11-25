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

See https://en.wikipedia.org/wiki/Golomb_ruler for more information.

For order 5: 2 solutions 0 1 4 9 11 ; 0 2 7 8 11   

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from sys import stdout


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Number of marks on the ruler
ORDER = 8


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Estimate an upper bound to the ruler length
MAX_LENGTH = (ORDER - 1) ** 2


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create array of variables corresponding to position ruler marks
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

print("\nSolving model....")
msol = mdl.solve(TimeLimit=100)

# Print solution
if msol:
    stdout.write("Solution: " + msol.get_solve_status() + "\n")
    stdout.write("Position of ruler marks: ")
    for v in marks:
        stdout.write(" " + str(msol[v]))
    stdout.write("\n")
    stdout.write("Solve time: " + str(round(msol.get_solve_time(), 2)) + "s\n")
else:
    stdout.write("Search status: " + msol.get_solve_status() + "\n")
