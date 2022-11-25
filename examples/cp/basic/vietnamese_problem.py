# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
Mathematical problem asked to 8 years old vietnamese children.

See: http://www.slate.fr/story/101809/puzzle-maths-vietnam for more information.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

# Create model
mdl = CpoModel()

# Create model variables
V = mdl.integer_var_list(size=9, min=1, max=9, name="V")
mdl.add(mdl.all_diff(V))
        
# Create constraint
mdl.add(V[0] + 13 * V[1] / V[2] + V[3] + 12 * V[4] - V[5] - 11 + V[6] * V[7] / V[8] - 10 == 66)


#-----------------------------------------------------------------------------
# Solve the model and display the result
#-----------------------------------------------------------------------------

# Solve model
print("Solving model....")
msol = mdl.solve(TimeLimit=10)

print("Solution: ")
msol.print_solution()
