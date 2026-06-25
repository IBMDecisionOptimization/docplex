# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------

# Given a set of locations J and a set of clients C
#  Minimize
#   sum(j in J) fixedCost[j]*used[j] +
#   sum(j in J)sum(c in C)cost[c][j]*supply[c][j]
#  Subject to
#   sum(j in J) supply[c][j] == 1                    for all c in C
#   sum(c in C) supply[c][j] <= (|C| - 1) * used[j]  for all j in J
#               supply[c][j] in { 0, 1 }             for all c in C, j in J
#                    used[j] in { 0, 1 }             for all j in J


import sys

from cplex.callbacks import UserCutCallback

from docplex.mp.callbacks.cb_mixin import *
from docplex.mp.model import Model

# Separate the disaggregated capacity constraints.
# In the model we have for each location j the constraint
#    sum(c in clients) supply[c][j] <= (nbClients-1) * used[j]
# Clearly, a client can only be serviced from a location that is used,
# so we also have a constraint
#    supply[c][j] <= used[j]
# that must be satisfied by every feasible solution. These constraints tend
# to be violated in LP relaxation. In this callback we separate them in cuts
# constraints added via a callback.


class CustomCutCallback(ConstraintCallbackMixin, UserCutCallback):
    # Callback constructor. Model object is set after registration.
    def __init__(self, env):
        UserCutCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.eps = 1e-6
        self.nb_cuts = 0

    def add_cut_constraint(self, ct):
        self.register_constraint(ct)

    @print_called("--> custom cut callback called: #{0}")
    def __call__(self):
        # fetch variable solution values at this point.
        sol = self.make_complete_solution()
        # fetch those constraints which are not satisfied.
        unsats = self.get_cpx_unsatisfied_cts(self.cts, sol, self.eps)
        for ct, cut, sense, rhs in unsats:
            # Method add() here is CPLEX's CutCallback.add()
            self.add(cut, sense, rhs)
            self.nb_cuts += 1
            print('-- add new cut[{0}]: [{1!s}]'.format(self.nb_cuts, ct))


def build_supply_model(fixed_costs, supply_costs, use_cuts=False, **kwargs):
    m = Model(name='suppy', **kwargs)

    nb_locations = len(fixed_costs)
    nb_clients = len(supply_costs)
    range_locations = range(nb_locations)
    range_clients = range(nb_clients)

    # --- Create variables. ---
    # - used[l]      If location l is used.
    # - supply[l][c] Amount shipped from location j to client c. This is a real
    #                number in [0,1] and specifies the percentage of c's
    #                demand that is served from location l.
    used = m.binary_var_list(range_locations, name='used')
    supply = m.continuous_var_matrix(range_clients, range_locations, lb=0, ub=1, name='supply')
    m.used = used
    m.supply = supply
    # --- add constraints ---
    # The supply for each client must sum to 1, i.e., the demand of each
    # client must be met.
    m.add_constraints(m.sum(supply[c, l] for l in range_locations) == 1 for c in range_clients)
    # Capacity constraint for each location. We just require that a single
    # location cannot serve all clients, that is, the capacity of each
    # location is nbClients-1. This makes the model a little harder to
    # solve and allows us to separate more cuts.
    m.add_constraints(
        m.sum(supply[c, l] for c in range_clients) <= (nb_clients - 1) * used[l] for l in range_locations)

    # Tweak some CPLEX parameters so that CPLEX has a harder time to
    # solve the model and our cut separators can actually kick in.
    # params = m.parameters
    # params.threads = 1
    # # params.mip.strategy.heuristicfreq = -1
    # params.mip.cuts.mircut = -1
    # params.mip.cuts.implied = -1
    # params.mip.cuts.gomory = -1
    # params.mip.cuts.flowcovers = -1
    # params.mip.cuts.pathcut = -1
    # params.mip.cuts.liftproj = -1
    # params.mip.cuts.zerohalfcut = -1
    # params.mip.cuts.cliques = -1
    # params.mip.cuts.covers = -1

    # --- set objective ---
    # objective is to minimize total cost, i.e. sum of location fixed cost and supply costs
    total_fixed_cost = m.dot(used, fixed_costs)
    m.add_kpi(total_fixed_cost, 'Total fixed cost')
    total_supply_cost = m.sum(supply[c, l] * supply_costs[c][l] for c in range_clients for l in range_locations)
    m.add_kpi(total_supply_cost, 'Total supply cost')
    m.minimize(total_fixed_cost + total_supply_cost)

    if use_cuts:
        # register a cut constraint callback
        # this links the model to the callback
        cut_cb = m.register_callback(CustomCutCallback)

        # store cut constraints inside the callback, as DOcplex objects
        # here we add the folwing cuts:
        #   supply[c,l] <= use[l] for c in clients, l in locations.
        # These constraints are implied by the capacity constraints, but might be violated in LP solutions.
        for l in range_locations:
            location_used = used[l]
            for c in range_clients:
                cut_cb.add_cut_constraint(supply[c, l] <= location_used)
        print('* add cut constraints callback with {0} cuts'.format(len(cut_cb.cts)))

    m.cut_callback = cut_cb
    return m


def print_solution(m, tol=1e-6):
    used = m.used
    supply = m.supply
    n_locations = len(used)
    n_clients = (int)(len(supply) / n_locations)
    for l in range(n_locations):
        if used[l].solution_value >= 1 - tol:
            print('Facility %d is used, it serves clients %s' %
                  (l, ', '.join((str(c) for c in range(n_clients) if supply[c, l].solution_value >= 1 - tol))))


# default data
DEFAULT_FIXED_COSTS = [480, 200, 320, 340, 300]
DEFAULT_SUPPLY_COSTS = [[24, 74, 31, 51, 84],
                        [57, 54, 86, 61, 68],
                        [57, 67, 29, 91, 71],
                        [54, 54, 65, 82, 94],
                        [98, 81, 16, 61, 27],
                        [13, 92, 34, 94, 87],
                        [54, 72, 41, 12, 78],
                        [54, 64, 65, 89, 89]]

def build_test_supply_model(use_cuts, **kwargs):
    return build_supply_model(DEFAULT_FIXED_COSTS, DEFAULT_SUPPLY_COSTS, use_cuts=use_cuts, **kwargs)

if __name__ == "__main__":
    # parse args
    args = sys.argv
    use_cuts = True
    for arg in args[1:]:
        if arg == '-cuts':
            use_cuts = False
        elif arg == '-nocuts':
            use_cuts = False
        else:
            print('Unknown argument %s' % arg)
    random = False
    if random:
        import numpy as np
        # trigger this to investigate higher volumes
        nl = 20
        nc = 100
        fixed = np.random.randint(100, high=500, size=nl)
        supply = np.random.randint(1, high=100, size=(nc, nl))
    else:
        fixed = DEFAULT_FIXED_COSTS
        supply = DEFAULT_SUPPLY_COSTS

    m = build_supply_model(fixed, supply, use_cuts=use_cuts)
    m.parameters.preprocessing.presolve = 0
    m.print_information()

    s = m.solve(log_output=False)
    assert s
    m.report()
    print_solution(m)
    # expected value is 843, regardless of cuts
    if not random:
        assert abs(m.objective_value - 843) <= 1e-4
    m.end()
