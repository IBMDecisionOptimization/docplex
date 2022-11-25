# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

"""
The problem consists in positioning different shapes in a larger shape (a square frame for instance)
by translating and rotating them in order to minimize the total interaction sum_ij 1/(1+d_ij^2).

In this example, the distance between objects is implemented as a blackbox function using the
module 'shapely' to compute the distance between shapes.

There are 3 decision variables attached to each shape x_i, y_i, r_i giving their position relative
to an initial position x0_i, y0_j, .

Because the shapes are completely different, we would have 6 arguments to the blackbox function: i, j, dx_ij, dy_ij, ri, rj.

Expression d_ij in the model would be something like: blackbox(d, i, j, x_i - x_j, y_i - y_j, ri, rj)
"""

import sys
from collections import namedtuple
try:
    from matplotlib import pyplot
    import shapely
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    from descartes   import PolygonPatch
except ImportError:
    print("Please ensure you have installed modules 'matplotlib', 'shapely' and 'descartes'.")
    sys.exit(0)


# Initialize data
#----------------

S = 60

Object = namedtuple('Object', 'name color geo')

Frame = Object("Frame", 'grey', Polygon([(-S/10,-S/10),(-S/10,S*11/10),(S*11/10,S*11/10),(S*11/10,-S/10)]).difference(Polygon([(0,0),(0,S),(S,S),(S,0)])))

Objects = [
  Object("Crescent", 'blue',  Point(20, 20).buffer(15).difference(Point(30, 20).buffer(17.5)).buffer(2)),
  Object("Bubble"  , 'gold',  unary_union([Point(10+7*i, 50).buffer(5) for i in range(4)])),
  Object("Triangle", 'red',   Polygon([(25, 10), (35, 30), (45, 10)]).buffer(1)),
  Object("Oval"    , 'green', Point(42, 40).buffer(15).intersection(Point(52, 40).buffer(15)).buffer(1))
]

AllObjects = [ Frame ] + Objects
NbObjects = len(AllObjects)

CX = [AllObjects[i].geo.centroid.x for i in range(NbObjects)]
CY = [AllObjects[i].geo.centroid.y for i in range(NbObjects)]


# Build the model
#----------------

from docplex.cp.model import *
import docplex.cp.solver.solver as solver

# check Solver version
sol_version = solver.get_solver_version()
if compare_natural(sol_version, '22.1') < 0:
    print("Blackbox functions are not implemented in this solver version: {}".format(sol_version))
    print("This example cannot be run.")
    sys.exit(0)

model = CpoModel()

# Decision variables: rotation r(i) of object i and translation (x[i],y[i])
x = [integer_var(min=-int(CX[i]), max=S-int(CX[i])) for i in range(NbObjects)]
y = [integer_var(min=-int(CY[i]), max=S-int(CY[i])) for i in range(NbObjects)]
r = [integer_var(min=0, max=359) for i in range(NbObjects)]

# In case we want to use the original solution as starting point
sp = CpoModelSolution()
for i in range(1, NbObjects):
    sp.add_integer_var_solution(x[i], 0)
    sp.add_integer_var_solution(y[i], 0)
    sp.add_integer_var_solution(r[i], 0)
model.set_starting_point(sp)

# Frame is fixed
model.add(x[0] == 0, y[0] == 0, r[0] == 0)

# Minimize interaction based on distances
def distance(i, j, dx, dy, ri, rj):
    """ Compute distance between two shapes
    :param i, j:    Indexes of the two shapes
    :param dx, dy:  Distance between shape i and j
    :param ri, rj:  Rotation angle of shapes i and j
    """
    obji = shapely.affinity.translate(shapely.affinity.rotate(AllObjects[i].geo, ri, origin='centroid'), dx, dy)
    objj = shapely.affinity.rotate(AllObjects[j].geo, rj, origin='centroid')
    return obji.distance(objj)
distance_bbx = CpoBlackboxFunction(distance)
dist2 = {(i, j) : distance_bbx(i, j, x[i]-x[j], y[i]-y[j], r[i], r[j])**2 for i in range(NbObjects) for j in range(i + 1, NbObjects)}

# An upper bound based on the distance between the centroids.
model.add(dist2[i, j] <= 2 * (S**2) for i in range(NbObjects) for j in range(i + 1, NbObjects))

# Add objective
interaction = sum( [1/(1+d) for d in dist2.values() ] )
model.add(minimize(interaction))


# Solve the model
# ---------------

print("Solving the model")
sol = model.solve(TimeLimit=30, trace_log=False)
if not sol:
    print("No solution found")
    sys.exit(0)

SolObjects = [Object(AllObjects[i].name, AllObjects[i].color,
                      shapely.affinity.translate(shapely.affinity.rotate(AllObjects[i].geo, sol[r[i]], origin='centroid'), sol[x[i]], sol[y[i]]))
              for i in range(NbObjects)]

print("Number of calls to the distance blackbox function: {}".format(distance_bbx.get_eval_count()))


# Display result
# --------------

import docplex.cp.utils_visu as visu
if not visu.is_visu_enabled():
    print("Visu is disabled.")
    sys.exit(0)

fig = pyplot.figure(1, figsize=(8,5), dpi=90)

# Add figure for initial state
ax = fig.add_subplot(121)
for i in range(NbObjects):
    patch = PolygonPatch(AllObjects[i].geo, fc=AllObjects[i].color, ec=AllObjects[i].color, alpha=0.5, zorder=2)
    ax.add_patch(patch)
ax.set_xlim(-S/10, S*11/10)
ax.set_ylim(-S/10, S*11/10)
ax.set_aspect("equal")

# Add figure for optimized state
ax = fig.add_subplot(122)
for i in range(NbObjects):
    patch = PolygonPatch(SolObjects[i].geo, fc=AllObjects[i].color, ec=AllObjects[i].color, alpha=0.5, zorder=2)
    ax.add_patch(patch)
ax.set_xlim(-S/10, S*11/10)
ax.set_ylim(-S/10, S*11/10)
ax.set_aspect("equal")

pyplot.show()
