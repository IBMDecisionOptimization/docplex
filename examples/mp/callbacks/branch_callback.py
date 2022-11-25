# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------


# This file shows how to connect CPLEX branch callbacks to a DOcplex model.
import math
import cplex
import cplex.callbacks as cpx_cb

from docplex.mp.callbacks.cb_mixin import *
from docplex.mp.model import Model
from collections import defaultdict, namedtuple


class MyBranch(ModelCallbackMixin, cpx_cb.BranchCallback):

    brtype_map = {'0': 'var', '1': 'sos1', '2': 'sos2', 'X': 'user'}
    def __init__(self, env):
        # non public...
        cpx_cb.BranchCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)
        self.nb_called = 0
        self.stats = defaultdict(int)

    def __call__(self):
        self.nb_called += 1
        br_type = self.get_branch_type()
        if (br_type == self.branch_type.SOS1 or
                br_type == self.branch_type.SOS2):
            return

        x = self.get_values()

        objval = self.get_objective_value()
        obj = self.get_objective_coefficients()
        feas = self.get_feasibilities()

        maxobj = -cplex.infinity
        maxinf = -cplex.infinity
        bestj = -1
        infeas = self.feasibility_status.infeasible

        for j in range(len(x)):
            if feas[j] == infeas:
                xj_inf = x[j] - math.floor(x[j])
                if xj_inf > 0.5:
                    xj_inf = 1.0 - xj_inf

                if (xj_inf >= maxinf and
                        (xj_inf > maxinf or abs(obj[j]) >= maxobj)):
                    bestj = j
                    maxinf = xj_inf
                    maxobj = abs(obj[j])

        if bestj < 0:
            return

        xj_lo = math.floor(x[bestj])
        # the (bestj, xj_lo, direction) triple can be any python object to
        # associate with a node
        dv = self.index_to_var(bestj)
        self.stats[dv] += 1
        # note that we convert the variable index to its docplex name
        print('---> BRANCH[{0}]---  custom branch callback, branch type is {1}, var={2!s}'
              .format(self.nb_called, self.brtype_map.get(br_type, '??'), dv))
        self.make_branch(objval, variables=[(bestj, "L", xj_lo + 1)],
                         node_data=(bestj, xj_lo, "UP"))
        self.make_branch(objval, variables=[(bestj, "U", xj_lo)],
                         node_data=(bestj, xj_lo, "DOWN"))

    def report(self, n=5):
        sorted_stats = sorted(self.stats.items(), key=lambda p: p[1], reverse=True)
        for k, (dv, occ) in enumerate(sorted_stats[:n], start=1):
            print('#{0} most branched: {1}, branched: {2}'.format(k, dv, occ))


def add_branch_callback(docplex_model, logged=False):
    # register a class callback once!!!
    bcb = docplex_model.register_callback(MyBranch)

    docplex_model.parameters.mip.interval = 1

    solution = docplex_model.solve(log_output=logged)
    assert solution is not None
    docplex_model.report()

    bcb.report(n=3)


Tdv = namedtuple('Tdv', ['dx', 'dy'])

neighbors = [Tdv(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]

assert len(neighbors) == 8

def build_lifegame_model(n, **kwargs):
    """ build a MIP model for a stable game of life configuration

    chessboard is (n+1) x (n+1)

    :param n:
    :return:
    """
    assert n >= 2

    assert Model.supports_logical_constraints(), "This model requires logical constraints cplex.version must be 12.80 or higher"
    lm = Model(name='game_of_life_{0}'.format(n), **kwargs)
    border = range(0, n + 2)
    inside = range(1, n + 1)

    # one binary var per cell
    life = lm.binary_var_matrix(border, border, name=lambda rc: 'life_%d_%d' % rc)

    # store sum of alive neighbors for interior cells
    sum_of_neighbors = {(i, j): lm.sum(life[i + n.dx, j + n.dy] for n in neighbors) for i in inside for j in inside}

    # all borderline cells are dead
    for j in border:
        life[0, j].ub = 0
        life[j, 0].ub = 0
        life[j, n + 1].ub = 0
        life[n + 1, j].ub = 0

    # ct1: the sum of alive neighbors for an alive cell is greater than 2
    for i in inside:
        for j in inside:
            lm.add(2 * life[i, j] <= sum_of_neighbors[i, j])

    # ct2: the sum of alive neighbors for an alive cell is less than 3
    for i in inside:
        for j in inside:
            lm.add(5 * life[i, j] + sum_of_neighbors[i, j] <= 8)

    # ct3: for a dead cell, the sum of alive neighbors cannot be 3
    for i in inside:
        for j in inside:
            ct3 = sum_of_neighbors[i, j] == 3
            lm.add(ct3 <= life[i, j])  # use logical cts here

    # satisfy the 'no 3 alive neighbors for extreme rows, columns
    for i in border:
        if i < n:
            for d in [1, n]:
                lm.add(life[i, d] + life[i + 1, d] + life[i + 2, d] <= 2)
                lm.add(life[d, i] + life[d, i + 1] + life[d, i + 2] <= 2)

    # symmetry breaking
    n2 = int(math.ceil(n/2))
    half1 = range(1, n2 + 1)
    half2 = range(n2 + 1, n)

    # there are more alive cells in left side
    lm.add(lm.sum(life[i1, j1] for i1 in half1 for j1 in inside) >= lm.sum(
        life[i2, j2] for i2 in half2 for j2 in inside))

    # there are more alive cells in upper side
    lm.add(lm.sum(life[i1, j1] for i1 in inside for j1 in half1) >= lm.sum(
        life[i2, j2] for i2 in inside for j2 in half2))

    # find maximum number of alive cells
    lm.maximize(lm.sum(life))

    # add a dummy kpi
    nlines = lm.sum( (lm.sum(life[i, j] for j in inside) >= 1) for i in inside)
    lm.add_kpi(nlines, 'nlines')

    # parameters: branch up, use heusristics, emphasis on opt, threads free
    lm.parameters.mip.strategy.branch = 1
    lm.parameters.mip.strategy.heuristicfreq = 10
    lm.parameters.emphasis.mip = 2
    lm.parameters.threads = 0

    # store data items as fields
    lm.size = n
    lm.life = life

    ini_s = lifegame_make_initial_solution(lm)
    if not ini_s.is_valid_solution():
        print('error in initial solution')
    else:
        lm.add_mip_start(ini_s)


    return lm


def lifegame_make_initial_solution(mdl):
    border3 = range(1, mdl.size-1, 3)
    life_vars = mdl.life
    vvmap = {}
    for i in border3:
        for j in border3:
            vvmap[life_vars[i, j]] = 1
            vvmap[life_vars[i+1, j]] = 1
            vvmap[life_vars[i, j+1]] = 1
            vvmap[life_vars[i+1, j+1]] = 1
    return mdl.new_solution(var_value_dict=vvmap)

if __name__ == "__main__":
    life_m = build_lifegame_model(n=6)
    add_branch_callback(life_m, logged=False)
    life_m.end()


