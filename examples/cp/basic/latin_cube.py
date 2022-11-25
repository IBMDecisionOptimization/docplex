# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
In combinatorics and in experimental design, a Latin cube is a 3 dimensions extension of the Latin square.

The latin cube is a n x n x n array filled with n different symbols,
each occurring exactly once in each row and exactly once in each column.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from sys import stdout


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Size of the cube
CUBE_SIZE = 4

# Indicate to constrain each square diagonal with all different symbols
CONSTRAIN_DIAGONALS = True


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create CPO model
mdl = CpoModel()

# Create grid of variables
GRNG = range(CUBE_SIZE)
grid = [[[mdl.integer_var(min=0, max=CUBE_SIZE - 1, name="C_{}_{}_{}".format(x, y, z)) for x in GRNG] for y in GRNG] for z in GRNG]

# Add constraints for each slice on direction x
for x in GRNG:
    # Add alldiff constraints for lines
    for l in GRNG:
        mdl.add(mdl.all_diff([grid[x][l][c] for c in GRNG]))

    # Add alldiff constraints for columns
    for c in GRNG:
        mdl.add(mdl.all_diff([grid[x][l][c] for l in GRNG]))

    # Add alldiff constraints for diagonals
    if CONSTRAIN_DIAGONALS:
        mdl.add(mdl.all_diff([grid[x][l][l] for l in GRNG]))
        mdl.add(mdl.all_diff([grid[x][l][CUBE_SIZE - l - 1] for l in GRNG]))

# Add constraints for each slice on direction y
for y in GRNG:
    # Add alldiff constraints for lines
    for l in GRNG:
        mdl.add(mdl.all_diff([grid[l][y][c] for c in GRNG]))

    # Add alldiff constraints for columns
    for c in GRNG:
        mdl.add(mdl.all_diff([grid[l][y][c] for l in GRNG]))

    # Add alldiff constraints for diagonals
    if CONSTRAIN_DIAGONALS:
        mdl.add(mdl.all_diff([grid[l][y][l] for l in GRNG]))
        mdl.add(mdl.all_diff([grid[l][y][CUBE_SIZE - l - 1] for l in GRNG]))

# Add constraints for each slice on direction z
for z in GRNG:
    # Add alldiff constraints for lines
    for l in GRNG:
        mdl.add(mdl.all_diff([grid[l][c][z] for c in GRNG]))

    # Add alldiff constraints for columns
    for c in GRNG:
        mdl.add(mdl.all_diff([grid[l][c][z] for l in GRNG]))

    # Add alldiff constraints for diagonals
    if CONSTRAIN_DIAGONALS:
        mdl.add(mdl.all_diff([grid[l][l][z] for l in GRNG]))
        mdl.add(mdl.all_diff([grid[l][CUBE_SIZE - l - 1][z] for l in GRNG]))

# Force first line to natural sequence
for c in GRNG:
    mdl.add(grid[0][0][c] == c)


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=10)

# Print solution
stdout.write("Solution:\n")
if msol:
    for x in GRNG:
        for l in GRNG:
            for c in GRNG:
                stdout.write(" " + chr(ord('A') + msol[grid[x][l][c]]))
            stdout.write('\n')
        stdout.write('\n')
else:
    stdout.write("No solution found\n")
