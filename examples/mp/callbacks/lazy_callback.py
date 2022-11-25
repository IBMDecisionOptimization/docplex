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
# Optionally, the capacity constraints can be separated from a lazy constraint
# callback instead of being stated as part of the initial model.
# See the usage message for how to switch between these options.


import sys

from cplex.callbacks import LazyConstraintCallback

from docplex.mp.callbacks.cb_mixin import *
from docplex.mp.model import Model


# Lazy constraint callback to enforce the capacity constraints.
# If used then the callback is invoked for every integer feasible solution
# CPLEX finds. For each location j it checks whether constraint
#    sum(c in C) supply[c][j] <= (|C| - 1) * used[j]
# is satisfied. If not then it adds the violated constraint as lazy constraint.
class CustomLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_lazy_cts = 0

    def add_lazy_constraints(self, cts):
        self.register_constraints(cts)

    @print_called('--> lazy constraint callback called: #{0}')
    def __call__(self):
        # fetch variable values into a solution
        sol = self.make_complete_solution()
        # for each lazy constraint, check whether it is verified,
        unsats = self.get_cpx_unsatisfied_cts(self.cts, sol, tolerance=1e-6)
        for ct, cpx_lhs, sense, cpx_rhs in unsats:
            self.add(cpx_lhs, sense, cpx_rhs)
            self.nb_lazy_cts += 1
            print('  -- new lazy constraint[{0}]: {1!s}'.format(self.nb_lazy_cts, ct))


def build_supply_model(fixed_costs, supply_costs, lazy=True, **kwargs):
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
    # in addition, a location cannot serve any client when not in use.
    # This constraint are delagated to the lazy constraint callback if the lazy flag is True.
    if not lazy:
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

    if lazy:
        # register a lazy constraint callback
        print('* add lazy constraints callback')
        lazyct_cb = m.register_callback(CustomLazyCallback)

        # store lazy constraints inside the callback, as DOcplex objects
        lazyct_cb.add_lazy_constraints(
            m.sum(supply[c, l] for c in range_clients) <= (nb_clients - 1) * used[l] for l in range_locations)
        print('* added lazy constraints callback with {0} constraints'.format(len(lazyct_cb.cts)))

    m.lazy_callback = lazyct_cb
    m.parameters.preprocessing.presolve = 0
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

def build_test_supply_model(lazy, **kwargs):
    return build_supply_model(DEFAULT_FIXED_COSTS, DEFAULT_SUPPLY_COSTS, lazy=lazy, **kwargs)


if __name__ == "__main__":
    # parse args
    args = sys.argv
    use_lazy = True
    for arg in args[1:]:
        if arg == '-lazy':
            use_lazy = True
        if arg == '-nolazy':
            use_lazy = True
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

    m = build_supply_model(fixed, supply, lazy=use_lazy)
    m.print_information()

    s = m.solve(log_output=True)
    assert s
    m.report()
    print_solution(m)
    # expected value is 843, regardless of using lazy constraints
    if not random:
        assert abs(m.objective_value - 843) <= 1e-4
    m.end()


# * model suppy solved with objective = 843.000
# *  KPI: Total fixed cost  = 520.000
# *  KPI: Total supply cost = 323.000
# Facility 1 is used, it serves clients 1, 3, 7
# Facility 2 is used, it serves clients 0, 2, 4, 5, 6
