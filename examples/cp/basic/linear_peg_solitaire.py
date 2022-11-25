# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
This game is a solitaire game whose objective is to exchange two groups or red and blue pegs
disposed linearly around a hole. The default game contains 2 pegs of each color, but may be extended
to more.

For two pegs of each color, the initial state is two red pegs and two blue pegs separated by a hole: RReBB.
Objective is to reach final state BBeRR using only the following allowed moves:
 - A red peg can step one position to the right into the empty space.
 - A red peg can jump two positions to the right into the empty space.
 - A blue peg can step one position to the left into the empty space.
 - A blue peg can jump two positions to the left into the empty space.

For example, for 2 pegs of each colors, the sequence of states is:
   RR.BB
   R.RBB
   RBR.B
   RBRB.
   RB.BR
   .BRBR
   B.RBR
   BBR.R
   BB.RR

More information on http://www.cems.uvm.edu/~rsnapp/teaching/cs32/lectures/pegsolitaire.pdf
(Robert Snapp, Department of Computer Science, University of Vermont)

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel
from sys import stdout


#-----------------------------------------------------------------------------
# Initialize the problem data
#-----------------------------------------------------------------------------

# Number of pegs of each color
NB_PEGS = 6


#-----------------------------------------------------------------------------
# Prepare the data for modeling
#-----------------------------------------------------------------------------

# Total number of holes (empty hole in the middle)
SIZE = 2 * NB_PEGS + 1

# Required number of moves (see reference document)
NB_MOVES = NB_PEGS * (NB_PEGS + 2)

# Integer values representing hole states
HOLE = 0
RED = 1
BLUE = 2

# Letters used to print the different pegs
PEG_LETTERS = ('.', 'R', 'B')


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create sequence of states. Each variable has 3 possible values: 0: Hole, 1: red peg, 2: blue peg
states = []
for s in range(NB_MOVES + 1):
    states.append(mdl.integer_var_list(SIZE, HOLE, BLUE, "State_" + str(s) + "_"))

# Create variables representing from index and to index for each move
fromIndex = mdl.integer_var_list(NB_MOVES, 0, SIZE - 1, "From_")
toIndex   = mdl.integer_var_list(NB_MOVES, 0, SIZE - 1, "To_")

# Add constraints between each state
for m in range(NB_MOVES):
    fvar = fromIndex[m]
    tvar = toIndex[m]
    fromState = states[m]
    toState = states[m + 1]
    # Constrain location of holes
    mdl.add(mdl.element(fromState, tvar) == HOLE)
    # Constrain move size and direction
    delta = tvar - fvar
    mdl.add(mdl.allowed_assignments(delta, [-2, -1, 1, 2]))
    peg = mdl.element(fromState, fvar)
    mdl.add( ((peg == RED) & (delta > 0)) | ((peg == BLUE) & (delta < 0)) )
    # Make moves
    mdl.add(mdl.element(toState, tvar) == mdl.element(fromState, fvar))
    mdl.add(mdl.element(toState, fvar) == HOLE)
    # Force equality of other positions
    for p in range(SIZE):
        mdl.add(mdl.if_then((p != fvar) & (p != tvar), fromState[p] == toState[p]))

# Set initial position
for p in range(NB_PEGS):
    mdl.add(states[0][p] == RED)
    mdl.add(states[0][p + NB_PEGS + 1] == BLUE)
mdl.add(states[0][NB_PEGS] == HOLE)

# Force last state to be final
expr = states[NB_MOVES][NB_PEGS] == HOLE
for p in range(NB_PEGS):
    expr &= states[NB_MOVES][p] == BLUE
    expr &= states[NB_MOVES][p + NB_PEGS + 1] == RED
mdl.add(expr)


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print("Solving model....")
msol = mdl.solve(TimeLimit=50)

# Print solution
if msol:
    print("Number of moves: " + str(NB_MOVES))
    for m in range(NB_MOVES + 1):
        rstr = "   " + ''.join(PEG_LETTERS[msol[states[m][x]]] for x in range(SIZE))
        if (m < NB_MOVES):
            rstr += "  {0} -> {1}".format(msol[fromIndex[m]], msol[toIndex[m]])
        print(rstr)
else:
    stdout.write("Solve status: " + msol.get_solve_status() + ", fail status: " + msol.get_fail_status() + "\n")
