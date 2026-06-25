# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
Sudoku is a logic-based, combinatorial number-placement puzzle.
The objective is to fill a 9x9 grid with digits so that each column, each row,
and each of the nine 3x3 sub-grids that compose the grid contains all of the digits from 1 to 9.
The puzzle setter provides a partially completed grid, which for a well-posed puzzle has a unique solution.

See https://en.wikipedia.org/wiki/Sudoku for details

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from sys import stdout


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Problem 1 (zero means cell to be filled with appropriate value)
SUDOKU_PROBLEM_1 = ( (0, 0, 0,  0, 9, 0,  1, 0, 0),
                     (2, 8, 0,  0, 0, 5,  0, 0, 0),
                     (7, 0, 0,  0, 0, 6,  4, 0, 0),

                     (8, 0, 5,  0, 0, 3,  0, 0, 6),
                     (0, 0, 1,  0, 0, 4,  0, 0, 0),
                     (0, 7, 0,  2, 0, 0,  0, 0, 0),

                     (3, 0, 0,  0, 0, 1,  0, 8, 0),
                     (0, 0, 0,  0, 0, 0,  0, 5, 0),
                     (0, 9, 0,  0, 0, 0,  0, 7, 0),
                   )

# Problem 2
SUDOKU_PROBLEM_2 = ( (0, 7, 0,  0, 0, 0,  0, 4, 9),
                     (0, 0, 0,  4, 0, 0,  0, 0, 0),
                     (4, 0, 3,  5, 0, 7,  0, 0, 8),

                     (0, 0, 7,  2, 5, 0,  4, 0, 0),
                     (0, 0, 0,  0, 0, 0,  8, 0, 0),
                     (0, 0, 4,  0, 3, 0,  5, 9, 2),

                     (6, 1, 8,  0, 0, 0,  0, 0, 5),
                     (0, 9, 0,  1, 0, 0,  0, 3, 0),
                     (0, 0, 5,  0, 0, 0,  0, 0, 7),
                  )


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create CPO model
mdl = CpoModel()

# Grid range
GRNG = range(9)

# Create grid of variables
grid = [[mdl.integer_var(min=1, max=9, name="C" + str(l) + str(c)) for l in GRNG] for c in GRNG]

# Add alldiff constraints for lines
for l in GRNG:
    mdl.add(mdl.all_diff([grid[l][c] for c in GRNG]))

# Add alldiff constraints for columns
for c in GRNG:
    mdl.add(mdl.all_diff([grid[l][c] for l in GRNG]))

# Add alldiff constraints for sub-squares
ssrng = range(0, 9, 3)
for sl in ssrng:
    for sc in ssrng:
        mdl.add(mdl.all_diff([grid[l][c] for l in range(sl, sl + 3) for c in range(sc, sc + 3)]))

# Initialize known cells
problem = SUDOKU_PROBLEM_2
for l in GRNG:
    for c in GRNG:
        v = problem[l][c]
        if v > 0:
            grid[l][c].set_domain((v, v))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def print_grid(grid):
    """ Print Sudoku grid """
    for l in GRNG:
        if (l > 0) and (l % 3 == 0):
           stdout.write('\n')
        for c in GRNG:
            v = grid[l][c]
            stdout.write('   ' if (c % 3 == 0) else ' ')
            stdout.write(str(v) if v > 0 else '.')
        stdout.write('\n')

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=10)

# Print solution
stdout.write("Initial problem:\n")
print_grid(problem)
stdout.write("Solution:\n")
if msol:
    sol = [[msol[grid[l][c]] for c in GRNG] for l in GRNG]
    print_grid(sol)
else:
    stdout.write("No solution found\n")
