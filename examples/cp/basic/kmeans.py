# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2021, 2022
# --------------------------------------------------------------------------

"""
K-means is a way of clustering points in a multi-dimensional space
where the set of points to be clustered are partitioned into k subsets.
The idea is to minimize the inter-point distances inside a cluster in
order to produce clusters which group together close points.

See https://en.wikipedia.org/wiki/K-means_clustering
"""


import numpy as np
from docplex.cp.model import CpoModel
import docplex.cp.solver.solver as solver
from docplex.cp.utils import compare_natural

def make_model(coords, k, trust_numerics=True):
    """
    Build a K-means model from a set of coordinate vectors (points),
    and a given number of clusters k.

    We assign each point to a cluster and minimize the objective which
    is the sum of the squares of the distances of each point to
    the centre of gravity of the cluster to which it belongs.

    Here, there are two ways of building the objective function.  One
    uses the sum of squares of the coordinates of points in a cluster
    minus the size of the cluster times the center value.  This is akin
    to the calculation of variance vi E[X^2] - E[X]^2.  This is the most
    efficient but can be numerically unstable due to massive cancellation.

    The more numerically stable (but less efficient) way to calculate the
    objective is the analog of the variance calculation (sum_i(X_i - mu_i)^2)/n
    """
    # Sizes and ranges
    n, d = coords.shape
    N, D, K = range(n), range(d), range(k)

    # Model, and decision variables.  x[c] = cluster to which node c belongs
    mdl = CpoModel()
    x = [mdl.integer_var(0, k-1, "C_{}".format(i)) for i in N]

    # Size (number of nodes) in each cluster.  If this is zero, we make
    # it 1 to avoid division by zero later (if a particular cluster is
    # not used).
    csize = [mdl.max(1, mdl.count(x, c)) for c in K]

    # Calculate total distance squared
    total_dist2 = 0
    for c in K:  # For each cluster
        # Boolean vector saying which points are in this cluster
        included = [x[i] == c for i in N]
        for dim in D:  # For each dimension
            # Points for each point in the given dimension (x, y, z, ...)
            point = coords[:, dim]

            # Calculate the cluster centre for this dimension
            centre = mdl.scal_prod(included, point) / csize[c]

            # Calculate the total distance^2 for this cluster & dimension
            if trust_numerics:
                sum_of_x2 = mdl.scal_prod(included, (p**2 for p in point))
                dist2 = sum_of_x2 - centre**2 * csize[c]
            else:
                all_dist2 = ((centre - p)**2 for p in point)
                dist2 = mdl.scal_prod(included, all_dist2)

            # Keep the total distance squared in a sum
            total_dist2 += dist2

    # Minimize the total distance squared
    mdl.minimize(total_dist2)
    return mdl


if __name__ == "__main__":
    import sys
    # Default values
    n, d, k, sd = 500, 2, 5, 1234

    # Accept number of points, number of dimensions, number of clusters, seed
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        d = int(sys.argv[2])
    if len(sys.argv) > 3:
        k = int(sys.argv[3])
    if len(sys.argv) > 4:
        sd = int(sys.argv[4])

    # Message
    print("Generating with N = {}, D = {}, K = {}".format(n, d, k))

    # Seed and generate coordinates on the unit hypercube
    np.random.seed(sd)
    coords = np.random.uniform(0, 1, size=(n, d))

    # Build model
    mdl = make_model(coords, k)

    # Solve using constraint programming
    mdl.solve(SearchType="Restart", TimeLimit=10, LogPeriod=50000)

    if compare_natural(solver.get_solver_version(), '22.1') >= 0:
        # Solve using neighborhood search
        mdl.solve(SearchType="Neighborhood", TimeLimit=10, LogPeriod=50000)
