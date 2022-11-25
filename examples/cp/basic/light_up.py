# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
Light Up, also called Akari, is a binary-determination logic puzzle published by Nikoli.

Light Up is played on a rectangular grid of white and black cells.
The player places light bulbs in white cells such that no two bulbs shine on each other,
until the entire grid is lit up. A bulb sends rays of light horizontally and vertically,
illuminating its entire row and column unless its light is blocked by a black cell.
A black cell may have a number on it from 0 to 4, indicating how many bulbs must be placed
adjacent to its four sides; for example, a cell with a 4 must have four bulbs around it,
one on each side, and a cell with a 0 cannot have a bulb next to any of its sides.
An unnumbered black cell may have any number of light bulbs adjacent to it, or none.
Bulbs placed diagonally adjacent to a numbered cell do not contribute to the bulb count.

See https://en.wikipedia.org/wiki/Light_Up_(puzzle).

Examples taken from https://www.brainbashers.com and https://en.wikipedia.org.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from sys import stdout


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Each problem is expressed as a list of strings, each one representing a line of the puzzle.
# Character may be:
#  - Blank for an empty cell
#  - A digit (in 0..4) for black cell that force a number of neighbor bulbs,
#  - Any character to represent a black cell

# Problem 1. Solution:
LIGHT_UP_PROBLEM_1 = ("  2  ",
                      "     ",
                      "  X0 ",
                      " 1   ",
                      "   1 ")

# Problem 2
LIGHT_UP_PROBLEM_2 = ("X  X     X",
                      "       X  ",
                      " 3    0   ",
                      "  2  X   1",
                      "   10X    ",
                      "    1XX   ",
                      "X   2  2  ",
                      "   X    X ",
                      "  1       ",
                      "0     1  0")

# Problem 2
LIGHT_UP_PROBLEM_3 = ("    X     X   1     ",
                      "3   X   2       X 2 ",
                      "   XX    1  XX      ",
                      " X 1X1  X1 3 X X2XX ",
                      "X      1X   X     0X",
                      "      X  XX XXXX    ",
                      "X       1  3        ",
                      "  X 3     X        X",
                      " 1X 1X X  X 2     X ",
                      " X    X   X   XXX XX",
                      " X X     0X0XX  1 X ",
                      " 0    0  0   X  X 1 ",
                      "        1       1 X ",
                      " X    X    X  1 X XX",
                      "X 2        2   X1X1 ",
                      "X   X 0 1   X  X  X ",
                      "   2     X     X 2  ",
                      "X       X   XX X    ",
                      "    0     2X      X ",
                      "XX  1   2  X   2X   ")

PUZZLE = LIGHT_UP_PROBLEM_3


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Retrieve size of the grid
WIDTH = len(PUZZLE[0])
HEIGHT = len(PUZZLE)

def get_neighbors(l, c):
    """ Build the list of neighbors of a given cell """
    res = []
    if c > 0:          res.append((l, c-1))
    if c < WIDTH - 1:  res.append((l, c+1))
    if l > 0:          res.append((l-1, c))
    if l < HEIGHT - 1: res.append((l+1, c))
    return res

def get_all_visible(l, c):
    """ Build the list of cells that are visible from a given one """
    res = [(l, c)]
    c2 = c - 1
    while c2 >= 0 and PUZZLE[l][c2] == ' ':
        res.append((l, c2))
        c2 -= 1
    c2 = c + 1
    while c2 < WIDTH and PUZZLE[l][c2] == ' ':
        res.append((l, c2))
        c2 += 1

    l2 = l - 1
    while l2 >= 0 and PUZZLE[l2][c] == ' ':
        res.append((l2, c))
        l2 -= 1
    l2 = l + 1
    while l2 < HEIGHT and PUZZLE[l2][c] == ' ':
        res.append((l2, c))
        l2 += 1
    return res

def get_right_empty_count(l, c):
    """ Get the number of empty cells at the right of the given one, including it """
    if PUZZLE[l][c] != ' ':
        return 1
    res = 1
    c += 1
    while c < WIDTH and PUZZLE[l][c] == ' ':
        c += 1
        res += 1
    return res

def get_down_empty_count(l, c):
    """ Get the number of empty cells at the down of the given one, including it """
    if PUZZLE[l][c] != ' ':
        return 1
    res = 1
    l += 1
    while l < HEIGHT and PUZZLE[l][c] == ' ':
        l += 1
        res += 1
    return res


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create CPO model
mdl = CpoModel()

# Create one binary variable for presence of bulbs in cells
bulbs = [[mdl.integer_var(min=0, max=1, name="C{}_{}".format(l, c)) for c in range(WIDTH)] for l in range(HEIGHT)]

# Force number of bulbs in black cells to zero
for l in range(HEIGHT):
    for c in range(WIDTH):
        if PUZZLE[l][c] != ' ':
            mdl.add(bulbs[l][c] == 0)

# Force number of bulbs around numbered cells
for l in range(HEIGHT):
    for c in range(WIDTH):
        v = PUZZLE[l][c]
        if v.isdigit():
            mdl.add(mdl.sum(bulbs[l2][c2] for l2, c2 in get_neighbors(l, c)) == int(v))

# Avoid multiple bulbs on adjacent empty cells
for l in range(HEIGHT):
    c = 0
    while c < WIDTH:
        nbc = get_right_empty_count(l, c)
        if nbc > 1:
            mdl.add(mdl.sum(bulbs[l][c2] for c2 in range(c, c + nbc)) <= 1)
        c += nbc
for c in range(WIDTH):
    l = 0
    while l < HEIGHT:
        nbc = get_down_empty_count(l, c)
        if nbc > 1:
            mdl.add(mdl.sum(bulbs[l2][c] for l2 in range(l, l + nbc)) <= 1)
        l += nbc

# Each empty cell must be lighten by at least one bulb
for l in range(HEIGHT):
    for c in range(WIDTH):
        if PUZZLE[l][c] == ' ':
            mdl.add(mdl.sum(bulbs[l2][c2] for l2, c2 in get_all_visible(l, c)) > 0)

# Minimize the total number of bulbs
nbbulbs = mdl.integer_var(0, HEIGHT * WIDTH, name="NbBulbs")
mdl.add(nbbulbs == sum(bulbs[l][c] for c in range(WIDTH) for l in range(HEIGHT)))
mdl.add(mdl.minimize(nbbulbs))


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

def print_grid(grid):
    """ Print grid """
    for l in grid:
        stdout.write('|')
        for v in l:
            stdout.write(" " + str(v))
        stdout.write(' |\n')

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=10)

# Print solution
stdout.write("Initial problem:\n")
print_grid(PUZZLE)
if msol:
    # Print solution grig
    psol = []
    stdout.write("Solution: (bulbs represented by *):\n")
    for l in range(HEIGHT):
        nl = []
        for c in range(WIDTH):
            if PUZZLE[l][c] == ' ':
                nl.append('*' if msol[bulbs[l][c]] > 0 else '.')
            else:
                nl.append(PUZZLE[l][c])
        psol.append(nl)
    print_grid(psol)
    print("Total bulbs: " + str(msol[nbbulbs]))
else:
    stdout.write("No solution found\n")
