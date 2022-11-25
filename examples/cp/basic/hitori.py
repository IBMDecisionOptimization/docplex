# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
Hitori is played with a grid of squares or cells, and each cell contains a
number. The objective is to eliminate numbers by filling in the squares such
that remaining cells do not contain numbers that appear more than once in
either a given row or column.

Filled-in cells cannot be horizontally or vertically adjacent, although they
can be diagonally adjacent. The remaining un-filled cells must form a single
component connected horizontally and vertically.

See https://en.wikipedia.org/wiki/Hitori

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from sys import stdout


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Problem 0 (for test). A solution is:
#   * 2 *
#   2 3 1
#   * 1 *
HITORI_PROBLEM_0 = ( (2, 2, 1),
                     (2, 3, 1),
                     (1, 1, 1),
                   )

# Problem 1. A solution is:
#   * 2 * 5 3
#   2 3 1 4 *
#   * 1 * 3 5
#   1 * 5 * 2
#   5 4 3 2 1
HITORI_PROBLEM_1 = ( (2, 2, 1, 5, 3),
                     (2, 3, 1, 4, 5),
                     (1, 1, 1, 3, 5),
                     (1, 3, 5, 4, 2),
                     (5, 4, 3, 2, 1),
                   )

# Problem 2. A solution is:
#   * 8 * 6 3 2 * 7
#   3 6 7 2 1 * 5 4
#   * 3 4 * 2 8 6 1
#   4 1 * 5 7 * 3 *
#   7 * 3 * 8 5 1 2
#   * 5 6 7 * 1 8 *
#   6 * 2 3 5 4 7 8
#   8 7 1 4 * 3 * 6
HITORI_PROBLEM_2 = ( (4, 8, 1, 6, 3, 2, 5, 7),
                     (3, 6, 7, 2, 1, 6, 5, 4),
                     (2, 3, 4, 8, 2, 8, 6, 1),
                     (4, 1, 6, 5, 7, 7, 3, 5),
                     (7, 2, 3, 1, 8, 5, 1, 2),
                     (3, 5, 6, 7, 3, 1, 8, 4),
                     (6, 4, 2, 3, 5, 4, 7, 8),
                     (8, 7, 1, 4, 2, 3, 5, 6),
                   )

# Problem 3, solution to discover !
HITORI_PROBLEM_3 = ( ( 2,  5,  6,  3,  8, 10,  7,  4, 13,  6, 14, 15,  9,  4,  1),
                     ( 3,  1,  7, 12,  8,  4, 10,  4,  4, 11,  5, 13,  4,  9,  2),
                     ( 4, 14, 10, 10, 14,  5, 11,  1,  6,  2,  7, 11, 13, 15, 12),
                     ( 5, 10,  2,  5, 13,  3,  8,  5,  9,  7,  4, 10,  6, 10,  2),
                     ( 1,  6,  8, 15, 10,  7,  4,  2, 15, 14,  9,  3,  3, 11,  4),
                     ( 6, 14,  3, 11,  2,  4,  9,  5,  7, 13, 12,  8, 10, 14,  1),
                     (12,  8, 14, 11,  3,  7, 15, 13, 10,  7, 12, 13,  5,  2, 13),
                     (11,  4, 12, 15,  5,  6,  5,  3, 15, 10,  7,  9,  5, 13, 14),
                     ( 8, 15,  4,  6, 15,  3, 13, 14,  6, 12, 10,  1, 11,  3,  5),
                     (15, 15,  9, 12,  1,  8, 11, 10,  2,  2, 11,  9,  4, 12,  2),
                     ( 7,  1,  9,  9, 10,  5,  3, 11, 13,  6,  7,  4, 12,  5,  8),
                     (14, 10, 13,  4, 12, 15, 11, 10,  5,  7,  8, 12,  5,  3,  6),
                     ( 5, 10, 11,  5, 11, 14, 14, 15,  8, 13, 13,  2,  7,  9,  9),
                     ( 9,  7, 15, 10, 12, 11,  8,  6,  1,  5,  7, 14, 13,  1,  3),
                     ( 6,  9,  1, 13,  6,  4, 12,  7, 14,  4,  2,  1,  3,  8, 12)
                   )


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

PUZZLE = HITORI_PROBLEM_3
SIZE = len(PUZZLE)


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

def get_neighbors(l, c):
    """ Build the list of neighbors of a given cell """
    res = []
    if c > 0:        res.append((l, c-1))
    if c < SIZE - 1: res.append((l, c+1))
    if l > 0:        res.append((l-1, c))
    if l < SIZE - 1: res.append((l+1, c))
    return res

# Create model
mdl = CpoModel()

# Create one binary variable for each colored cell
color = [[mdl.integer_var(min=0, max=1, name="C" + str(l) + "_" + str(c)) for c in range(SIZE)] for l in range(SIZE)]

# Forbid adjacent colored cells
for l in range(SIZE):
    for c in range(SIZE - 1):
        mdl.add((color[l][c] + color[l][c + 1]) < 2)
for c in range(SIZE):
    for l in range(SIZE - 1):
        mdl.add((color[l][c] + color[l + 1][c]) < 2)

# Color cells for digits occurring more than once
for l in range(SIZE):
    lvals = []  # List of values already processed
    for c in range(SIZE):
        v = PUZZLE[l][c]
        if v not in lvals:
            lvals.append(v)
            lvars = [color[l][c]]
            for c2 in range(c + 1, SIZE):
                if PUZZLE[l][c2] == v:
                    lvars.append(color[l][c2])
            # Add constraint if more than one occurrence of the value
            nbocc = len(lvars)
            if nbocc > 1:
                mdl.add(mdl.sum(lvars) >= nbocc - 1)
for c in range(SIZE):
    lvals = []  # List of values already processed
    for l in range(SIZE):
        v = PUZZLE[l][c]
        if v not in lvals:
            lvals.append(v)
            lvars = [color[l][c]]
            for l2 in range(l + 1, SIZE):
                if PUZZLE[l2][c] == v:
                    lvars.append(color[l2][c])
            # Add constraint if more than one occurrence of the value
            nbocc = len(lvars)
            if nbocc > 1:
                mdl.add(mdl.sum(lvars) >= nbocc - 1)

# Each cell (blank or not) must be adjacent to at least another
for l in range(SIZE):
    for c in range(SIZE):
        lvars = [color[l2][c2] for l2, c2 in get_neighbors(l, c)]
        mdl.add(mdl.sum(lvars) < len(lvars))

# At least cell 0,0 or cell 0,1 is blank.
# Build table of distance to one of these cells
# Black cells are associated to a max distance SIZE*SIZE
MAX_DIST = SIZE * SIZE
distance = [[mdl.integer_var(min=0, max=MAX_DIST, name="D" + str(l) + "_" + str(c)) for c in range(SIZE)] for l in range(SIZE)]
mdl.add(distance[0][0] == mdl.conditional(color[0][0], MAX_DIST, 0))
mdl.add(distance[0][1] == mdl.conditional(color[0][1], MAX_DIST, 0))
for c in range(2, SIZE):
    mdl.add( distance[0][c] == mdl.conditional(color[0][c], MAX_DIST, 1 + mdl.min(distance[l2][c2] for l2, c2 in get_neighbors(0, c))) )
for l in range(1, SIZE):
    for c in range(SIZE):
        mdl.add( distance[l][c] == mdl.conditional(color[l][c], MAX_DIST, 1 + mdl.min(distance[l2][c2] for l2, c2 in get_neighbors(l, c))) )

# Force distance of blank cells to be less than max
for l in range(SIZE):
    for c in range(SIZE):
        mdl.add((color[l][c] > 0) | (distance[l][c] < MAX_DIST))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def print_grid(grid):
    """ Print Hitori grid """
    mxlen = max([len(str(grid[l][c])) for l in range(SIZE) for c in range(SIZE)])
    frmt = " {:>" + str(mxlen) + "}"
    for l in grid:
        for v in l:
            stdout.write(frmt.format(v))
        stdout.write('\n')

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=100)

# Print solution
stdout.write("Initial problem:\n")
print_grid(PUZZLE)
stdout.write("Solution:\n")
if msol:
    # Print solution grig
    psol = []
    for l in range(SIZE):
        nl = []
        for c in range(SIZE):
            nl.append('.' if msol[color[l][c]] > 0 else PUZZLE[l][c])
        psol.append(nl)
    print_grid(psol)
    # Print distance grid
    print("Distances:")
    psol = [['.' if msol[distance[l][c]] == MAX_DIST else msol[distance[l][c]] for c in range(SIZE)] for l in range(SIZE)]
    print_grid(psol)
else:
    stdout.write("No solution found\n")
