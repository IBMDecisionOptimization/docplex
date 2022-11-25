# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------

# This file shows how to connect CPLEX incumbent callbacks to a DOcplex model.
# It is inspired from a blog article on Linear Programming: http://brg.a2hosted.com/?page_id=1316

import cplex.callbacks as cpx_cb
import cplex._internal._constants as cpxcst

from docplex.mp.callbacks.cb_mixin import *
from docplex.mp.model import Model


class CustomIncumbentCallback(ModelCallbackMixin, cpx_cb.IncumbentCallback):

    incumbent_sources = {
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_NODESOLN: 'node',
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_HEURSOLN: 'heuristic',
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_USERSOLN: 'user',
        cpxcst.CPX_CALLBACK_MIP_INCUMBENT_MIPSTART: 'mipstart'
    }

    def __init__(self, env):
        # non public...
        cpx_cb.IncumbentCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)
        self.nb_incumbents = 0

    def __call__(self):
        self.nb_incumbents += 1
        obj = self.get_objective_value()
        src = self.get_solution_source()
        src_name = self.incumbent_sources.get(src, ' unknown???')
        print('{0}> found incumbent with objective value {1}, coming from: {2}'.format(self.nb_incumbents, obj, src_name))


# the circles are numbered as follows:
#
# 1               (1, 1)
# 2   3         (2, 1)   (2, 2)
# 4   5   6   (3, 1)   (3, 2)   (3, 3)
# ....
def build_hearts(r, **kwargs):
    # initialize the model
    mdl = Model('love_hearts_%d' % r, **kwargs)

    # the dictionary of decision variables, one variable
    # for each circle with i in (1 .. r) as the row and
    # j in (1 .. i) as the position within the row
    idx = [(i, j) for i in range(1, r + 1) for j in range(1, i + 1)]
    a = mdl.binary_var_dict(idx, name=lambda idx_tuple: "a_%d_%d" % (idx_tuple[0], idx_tuple[1]))

    # the constraints - enumerate all equilateral triangles
    # and prevent any such triangles being formed by keeping
    # the number of included circles at its vertexes below 3

    # for each row except the last
    for i in range(1, r):
        # for each position in this row
        for j in range(1, i + 1):
            # for each triangle of side length (k) with its upper vertex at
            # (i, j) and its sides parallel to those of the overall shape
            for k in range(1, r - i + 1):
                # the sets of 3 points at the same distances clockwise along the
                # sides of these triangles form k equilateral triangles
                for m in range(k):
                    u, v, w = (i + m, j), (i + k, j + m), (i + k - m, j + k - m)
                    mdl.add_constraint(a[u] + a[v] + a[w] <= 2)

    mdl.maximize(mdl.sum(a))
    return mdl


if __name__ == "__main__":
    love = build_hearts(r=11)
    love.register_callback(CustomIncumbentCallback)

    love.parameters.mip.interval = 1

    s10 = love.solve(log_output=False)
    assert s10 is not None
    love.report()
    love.end()

