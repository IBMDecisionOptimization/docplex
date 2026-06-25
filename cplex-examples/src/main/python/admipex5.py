#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: admipex5.py
 
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Solve a facility location problem with cut callbacks or lazy constraints.

See admipex8.py for an implementation using the more recent generic
callback.

Given a set of locations J and a set of clients C, the following model is
solved:

 Minimize
  sum(j in J) fixedCost[j]*used[j] +
  sum(j in J)sum(c in C) cost[c][j]*supply[c][j]
 Subject to
  sum(j in J) supply[c][j] == 1                    for all c in C
  sum(c in C) supply[c][j] <= (|C| - 1) * used[j]  for all j in J
              supply[c][j] in {0, 1}               for all c in C, j in J
                   used[j] in {0, 1}               for all j in J

In addition to the constraints stated above, the code also separates
a disaggregated version of the capacity constraints (see comments for the
cut callback) to improve performance.

Optionally, the capacity constraints can be separated from a lazy
constraint callback instead of being stated as part of the initial model.

See the usage message for how to switch between these options.
"""
import sys

import cplex
from cplex.callbacks import UserCutCallback, LazyConstraintCallback
from inputdata import read_dat_file

# epsilon used for violation of cuts
EPS = 1e-6


def usage(name):
    """Prints a usage statement."""
    msg = """Usage: %s [options...]
 By default, a user cut callback is used to dynamically
 separate constraints.

 Supported options are:
 -table       Instead of the default behavior, use a
              static table that holds all cuts and
              scan that table for violated cuts.
 -no-cuts     Do not separate any cuts.
 -lazy        Do not include capacity constraints in the
              model. Instead, separate them from a lazy
              constraint callback.
 -data=<dir>  Specify the directory in which the data
              file facility.dat is located.
"""
    print(msg % name)
    sys.exit(2)


class Disaggregated(UserCutCallback):
    """Separate the disaggregated capacity constraints.

    In the model we have for each location j the constraint

    sum(c in clients) supply[c][j] <= (nbClients-1) * used[j]

    Clearly, a client can only be serviced from a location that is used,
    so we also have a constraint

    supply[c][j] <= used[j]

    that must be satisfied by every feasible solution. These constraints tend
    to be violated in LP relaxation. In this callback we separate them.
    """

    # Callback constructor. Fields 'locations', 'clients', 'used', 'supply'
    # are set externally after registering the callback.
    def __init__(self, env):
        super().__init__(env)

    def __call__(self):
        for j in self.locations:
            for c in self.clients:
                s, o = self.get_values([self.supply[c][j], self.used[j]])
                if s > o + EPS:
                    print('Adding supply(%d)(%d) <= used(%d) [%f > %f]' %
                          (c, j, j, s, o))
                    self.add(cut=cplex.SparsePair([self.supply[c][j], self.used[j]],
                                                  [1.0, -1.0]),
                             sense='L', rhs=0.0)


class CutsFromTable(UserCutCallback):
    """Variant of the Disaggregated callback that does not look for
    violated cuts dynamically. Instead it uses a static table of cuts and
    scans this table for violated cuts.
    """

    # Callback constructor. The fields 'cutlhs' and 'cutrhs' are set externally
    # after the callback is registered.
    def __init__(self, env):
        super().__init__(env)

    def __call__(self):
        for lhs, rhs in zip(self.cutlhs, self.cutrhs):
            # Compute activity of left-hand side
            act = sum(c * x for c, x in zip(lhs.val, self.get_values(lhs.ind)))
            if act > rhs + EPS:
                print('Adding %s [act = %f]' % (str(lhs), act))
                self.add(cut=lhs, sense="L", rhs=rhs)


class LazyCallback(LazyConstraintCallback):
    """Lazy constraint callback to enforce the capacity constraints.

    If used then the callback is invoked for every integer feasible
    solution CPLEX finds. For each location j it checks whether
    constraint

    sum(c in C) supply[c][j] <= (|C| - 1) * used[j]

    is satisfied. If not then it adds the violated constraint as lazy
    constraint.
    """

    # Callback constructor. Fields 'locations', 'clients', 'used', 'supply'
    # are set externally after registering the callback.
    def __init__(self, env):
        super().__init__(env)

    def __call__(self):
        for j in self.locations:
            isused = self.get_values(self.used[j])
            served = sum(self.get_values(
                [self.supply[c][j] for c in self.clients]))
            if served > (len(self.clients) - 1.0) * isused + EPS:
                print('Adding lazy constraint %s <= %d*used(%d)' %
                      (' + '.join(['supply(%d)(%d)' % (x, j) for x in self.clients]),
                       len(self.clients) - 1, j))
                self.add(constraint=cplex.SparsePair(
                    [self.supply[c][j] for c in self.clients] + [self.used[j]],
                    [1.0] * len(self.clients) + [-(len(self.clients) - 1)]),
                    sense='L',
                    rhs=0.0)


def admipex5(args):
    # Set default arguments and parse command line
    datadir = '../../../examples/data'
    from_table = False
    lazy = False
    use_callback = True
    for arg in args[1:]:
        if arg.startswith('-data='):
            datadir = arg[6:]
        elif arg == '-table':
            from_table = True
        elif arg == '-lazy':
            lazy = True
        elif arg == '-no-cuts':
            use_callback = False
        else:
            print('Unknown argument %s' % arg)
            usage(args[0])

    # Read in data file. If no file name is given on the command line
    # we use a default file name. The data we read is
    # fixedcost  -- a list/array of facility fixed cost
    # cost       -- a matrix for the costs to serve each client by each
    #               facility
    fixedcost, cost, _ = read_dat_file(datadir + '/' + 'facility.dat')

    # Create the model
    locations = list(range(len(fixedcost)))
    clients = list(range(len(cost)))
    cpx = cplex.Cplex()
    # Create variables.
    # - used[j]      If location j is used.
    # - supply[c][j] Amount shipped from location j to client c. This is a
    #                number in [0,1] and specifies the percentage of c's
    #                demand that is served from location i.
    # Note that we also create the objective function along with the variables
    # by specifying the objective coefficient for each variable in the 'obj'
    # argument.
    used = cpx.variables.add(obj=fixedcost,
                             lb=[0] * len(locations), ub=[1] * len(locations),
                             types=['B'] * len(locations),
                             names=['used(%d)' % (j) for j in locations])
    supply = [cpx.variables.add(obj=[cost[c][j] for j in locations],
                                lb=[0] * len(locations), ub=[1] * len(locations),
                                types=['B'] * len(locations),
                                names=['supply(%d)(%d)' % (c, j) for j in locations])
              for c in clients]

    # The supply for each client must sum to 1, i.e., the demand of each
    # client must be met.
    cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(supply[c],
                                                          [1.0] * len(supply[c]))
                                         for c in clients],
                               senses=['E'] * len(clients),
                               rhs=[1.0] * len(clients))

    # Capacity constraint for each location. We just require that a single
    # location cannot serve all clients, that is, the capacity of each
    # location is nbClients-1. This makes the model a little harder to
    # solve and allows us to separate more cuts.
    if not lazy:
        cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(
            [supply[c][j] for c in clients] + [used[j]],
            [1.0] * len(clients) + [-(len(clients) - 1.0)])
            for j in locations],
            senses=['L'] * len(locations),
            rhs=[0] * len(locations))

    # Tweak some CPLEX parameters so that CPLEX has a harder time to
    # solve the model and our cut separators can actually kick in.
    cpx.parameters.threads.set(1)
    cpx.parameters.mip.strategy.heuristicfreq.set(-1)
    cpx.parameters.mip.cuts.mircut.set(-1)
    cpx.parameters.mip.cuts.implied.set(-1)
    cpx.parameters.mip.cuts.gomory.set(-1)
    cpx.parameters.mip.cuts.flowcovers.set(-1)
    cpx.parameters.mip.cuts.pathcut.set(-1)
    cpx.parameters.mip.cuts.liftproj.set(-1)
    cpx.parameters.mip.cuts.zerohalfcut.set(-1)
    cpx.parameters.mip.cuts.cliques.set(-1)
    cpx.parameters.mip.cuts.covers.set(-1)

    if use_callback:
        if from_table:
            # Generate all disaggregated constraints and put them into a
            # table that is scanned by the callback.
            usercb = cpx.register_callback(CutsFromTable)
            usercb.cutlhs = [cplex.SparsePair([supply[c][j], used[j]],
                                              [1.0, -1.0])
                             for j in locations for c in clients]
            usercb.cutrhs = [0] * len(locations) * len(clients)
        else:
            usercb = cpx.register_callback(Disaggregated)
            usercb.clients = clients
            usercb.locations = locations
            usercb.used = used
            usercb.supply = supply
    if lazy:
        lazycb = cpx.register_callback(LazyCallback)
        lazycb.clients = clients
        lazycb.locations = locations
        lazycb.used = used
        lazycb.supply = supply

    cpx.write('model.lp')
    cpx.solve()

    print('Solution status:                   %d' % cpx.solution.get_status())
    print('Nodes processed:                   %d' %
          cpx.solution.progress.get_num_nodes_processed())
    print('Active user cuts/lazy constraints: %d' %
          cpx.solution.MIP.get_num_cuts(cpx.solution.MIP.cut_type.user))
    tol = cpx.parameters.mip.tolerances.integrality.get()
    print('Optimal value:                     %f' %
          cpx.solution.get_objective_value())
    values = cpx.solution.get_values()
    for j in [x for x in locations if values[used[x]] >= 1 - tol]:
        print('Facility %d is used, it serves clients %s' %
              (j, ', '.join([str(x) for x in clients
                             if values[supply[x][j]] >= 1 - tol])))



if __name__ == "__main__":
    admipex5(sys.argv)
