# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------

#  heuristic_callback.py -  Use the heuristic callback
#                          for optimizing a MIP problem
#
# To run this example, the user must specify a problem file.
#
# You can run this example at the command line by
#
#    python heuristic_callback.py <filename>

import sys

from cplex.callbacks import HeuristicCallback

from docplex.mp.callbacks.cb_mixin import *


class RoundDown(ModelCallbackMixin, HeuristicCallback):
    def __init__(self, env):
        HeuristicCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)

    @print_called('--> calling my_round_down callback... #{0}')
    def __call__(self):
        feas = self.get_feasibilities()
        var_indices = [j for j, f in enumerate(feas) if f == self.feasibility_status.feasible]
        if var_indices:
            # this shows how to get back to the DOcplex variable from the index
            # but is not necessary for the logic.
            dvars = [self.index_to_var(v) for v in var_indices]
            print('* rounded vars = [{0}]'.format(', '.join([v.name for v in dvars[:3]])))
            # -- calling set-solution in cplex callback class
            self.set_solution([var_indices, [0.0] * len(var_indices)])


def try_heuristic_cb_on_model(mdl):
    mdl.register_callback(RoundDown)
    # tweak cplex parameters
    mdl.parameters.mip.tolerances.mipgap = 1.0e-6
    mdl.parameters.mip.strategy.search = 0

    s = mdl.solve()
    assert s is not None

    mdl.report()
    return s


def try_heuristic_cb_on_file(filename):
    from docplex.mp.model_reader import ModelReader
    mdl = ModelReader.read(filename)
    if mdl:
        return mdl, try_heuristic_cb_on_model(mdl)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        data_file = "data/location.lp"
        expected = 499
    elif len(sys.argv) == 2:
        data_file = sys.argv[1]
        expected = None
    else:
        print("Usage: heuristic_callback.py filename")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        sys.exit(-1)
    mdl, s = try_heuristic_cb_on_file(data_file)
    if expected:
        assert abs(s.objective_value - expected) <= 1
    mdl.end()
