# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
In combinatorics and in experimental design, a Latin square is an n x n array filled with n different symbols,
each occurring exactly once in each row and exactly once in each column.
Here is an example:

 A B C D
 D C B A
 B A D C
 C D A B

More information is available on https://en.wikipedia.org/wiki/Latin_square

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from sys import stdout


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Size of the square
SQUARE_SIZE = 16


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create CPO model
mdl = CpoModel()

# Create grid of variables
GRNG = range(SQUARE_SIZE)
grid = [[mdl.integer_var(min=0, max=SQUARE_SIZE - 1, name="C_{}_{}".format(l, c)) for l in GRNG] for c in GRNG]

# Add alldiff constraints for lines
for l in GRNG:
    mdl.add(mdl.all_diff([grid[l][c] for c in GRNG]))

# Add alldiff constraints for columns
for c in GRNG:
    mdl.add(mdl.all_diff([grid[l][c] for l in GRNG]))

# Add alldiff constraints for diagonals
mdl.add(mdl.all_diff([grid[l][l] for l in GRNG]))
mdl.add(mdl.all_diff([grid[l][SQUARE_SIZE - l - 1] for l in GRNG]))

# Force first line to natural sequence
for c in GRNG:
    mdl.add(grid[0][c] == c)


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=10)

# Print solution
stdout.write("Solution:\n")
if msol:
    for l in GRNG:
        for c in GRNG:
            stdout.write(" " + chr(ord('A') + msol[grid[l][c]]))
        stdout.write('\n')
else:
    stdout.write("No solution found\n")
