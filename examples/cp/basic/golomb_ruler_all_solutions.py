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

This implementation differs from the 'basic' implementation, given in the
examp;e module golomb_ruler.py, because it calls the solver twice:
 * First time to know the minimal size of the ruler for the required order,
 * A second time to list all possible rulers for this optimal size

See https://en.wikipedia.org/wiki/Golomb_ruler for more information.

For order 5: 2 solutions: 0 1 4 9 11
                          0 2 7 8 11
For order 7: 6 solutions: 0 1 4 10 18 23 25
                          0 1 7 11 20 23 25
                          0 1 11 16 19 23 25
                          0 2 3 10 16 21 25
                          0 2 7 13 21 22 25

Please refer to documentation for appropriate setup of solving configuration.
"""


from docplex.cp.model import CpoModel
from docplex.cp.utils import CpoNotSupportedException
from sys import stdout

#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Number of marks on the ruler
ORDER = 7


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
minexpr = mdl.minimize(marks[ORDER - 1])
mdl.add(minexpr)


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# First solve the model to find the smallest ruler length
msol = mdl.solve(TimeLimit=100)
if not msol:
    print("No Golomb ruler available for order " + str(ORDER))
else:
    rsize = msol[marks[ORDER - 1]]
    print("Shortest ruler for order " + str(ORDER) + " has length " + str(rsize))
    # Remove minimization from the model
    mdl.remove(minexpr)
    # Force position of last mark
    mdl.add(marks[ORDER - 1] == rsize)

    # Request all solutions
    print("List of all possible rulers for length {}:".format(rsize))
    siter = mdl.start_search(SearchType='DepthFirst', Workers=1, TimeLimit=100) # Parameters needed to avoid duplicate solutions
    try:
        for i, msol in enumerate(siter):
            stdout.write(str(i + 1) + ": ")
            for v in marks:
                stdout.write(" " + str(msol[v]))
            stdout.write("\n")
    except CpoNotSupportedException:
        print("This instance of the solver does not support solution iteration.")
