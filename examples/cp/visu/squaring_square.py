# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The aim of the square example is to place a set of small squares of
different sizes into a large square.

See https://en.wikipedia.org/wiki/Squaring_the_square for details on this classical problem.

This version is extended and uses matplotlib to draw the result at the end.
Requires installation of numpy (installer) and following python packages:
    "pip install matplotlib python-dateutil pyparsing"

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import *


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Size of the englobing square
SIZE_SQUARE = 112

# Sizes of the sub-squares
SIZE_SUBSQUARE = [50, 42, 37, 35, 33, 29, 27, 25, 24, 19, 18, 17, 16, 15, 11, 9, 8, 7, 6, 4, 2]
NB_SUBSQUARE = len(SIZE_SUBSQUARE)

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create array of variables for subsquares
vx = [interval_var(size=SIZE_SUBSQUARE[i], name='X{}'.format(i), end=(0, SIZE_SQUARE)) for i in range(NB_SUBSQUARE)]
vy = [interval_var(size=SIZE_SUBSQUARE[i], name='Y{}'.format(i), end=(0, SIZE_SQUARE)) for i in range(NB_SUBSQUARE)]

# Create dependencies between variables
for i in range(len(SIZE_SUBSQUARE)):
    for j in range(i):
        mdl.add(  (end_of(vx[i]) <= start_of(vx[j])) | (end_of(vx[j]) <= start_of(vx[i]))
                | (end_of(vy[i]) <= start_of(vy[j])) | (end_of(vy[j]) <= start_of(vy[i])))

# To speed-up the search, create cumulative expressions on each dimension
rx = sum([pulse(vx[i], SIZE_SUBSQUARE[i]) for i in range(NB_SUBSQUARE)])
mdl.add(always_in(rx, (0, SIZE_SQUARE), SIZE_SQUARE, SIZE_SQUARE))

ry = sum([pulse(vy[i], SIZE_SUBSQUARE[i]) for i in range(NB_SUBSQUARE)])
mdl.add(always_in(ry, (0, SIZE_SQUARE), SIZE_SQUARE, SIZE_SQUARE))

# Define search phases, also to speed-up the search
mdl.set_search_phases([search_phase(vx), search_phase(vy)])


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print('Solving model...')
res = mdl.solve(TimeLimit=20, LogPeriod=50000)
print('Solution: ')
res.print_solution()

import docplex.cp.utils_visu as visu
if res and visu.is_visu_enabled():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Polygon

    # Plot external square
    print('Plotting squares...')
    fig, ax = plt.subplots()
    plt.plot((0, 0), (0, SIZE_SQUARE), (SIZE_SQUARE, SIZE_SQUARE), (SIZE_SQUARE, 0))
    for i in range(len(SIZE_SUBSQUARE)):
        # Display square i
        sx, sy = res.get_var_solution(vx[i]), res.get_var_solution(vy[i])
        (sx1, sx2, sy1, sy2) = (sx.get_start(), sx.get_end(), sy.get_start(), sy.get_end())
        poly = Polygon([(sx1, sy1), (sx1, sy2), (sx2, sy2), (sx2, sy1)], fc=cm.Set2(float(i) / len(SIZE_SUBSQUARE)))
        ax.add_patch(poly)
        # Display identifier of square i at its center
        ax.text(float(sx1 + sx2) / 2, float(sy1 + sy2) / 2, str(SIZE_SUBSQUARE[i]), ha='center', va='center')
    plt.margins(0)
    plt.show()
