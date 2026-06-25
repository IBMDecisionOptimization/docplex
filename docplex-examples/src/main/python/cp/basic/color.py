# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The problem involves choosing colors for the countries on a map in
such a way that at most four colors (blue, white, yellow, green) are
used and no neighboring countries are the same color. In this exercise,
you will find a solution for a map coloring problem with six countries:
Belgium, Denmark, France, Germany, Luxembourg, and the Netherlands.

Please refer to documentation for appropriate setup of solving configuration.
"""

from docplex.cp.model import CpoModel

# Create CPO model
mdl = CpoModel()

# Create model variables containing colors of the countries
Belgium     = mdl.integer_var(0, 3, "Belgium")
Denmark     = mdl.integer_var(0, 3, "Denmark")
France      = mdl.integer_var(0, 3, "France")
Germany     = mdl.integer_var(0, 3, "Germany")
Luxembourg  = mdl.integer_var(0, 3, "Luxembourg")
Netherlands = mdl.integer_var(0, 3, "Netherlands")
ALL_COUNTRIES = (Belgium, Denmark, France, Germany, Luxembourg, Netherlands)
        
# Create constraints
mdl.add(Belgium != France)
mdl.add(Belgium != Germany)
mdl.add(Belgium != Netherlands)
mdl.add(Belgium != Luxembourg)
mdl.add(Denmark != Germany)
mdl.add(France  != Germany)
mdl.add(France  != Luxembourg)
mdl.add(Germany != Luxembourg)
mdl.add(Germany != Netherlands)

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=10)

if msol:
    print("Solution status: " + msol.get_solve_status())
    colors = ("Yellow", "Red", "Green", "Blue")
    for country in ALL_COUNTRIES:
        print("   " + country.get_name() + ": " + colors[msol[country]])
else:
    print("No solution found")

# Print solver log
# print("\nSolver log:")
# print(msol.get_solver_log())
