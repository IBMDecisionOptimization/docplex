#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: admipex8.py
 
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
Solve a capacitated facility location problem with cutting planes using
the new generic callback interface.

See admipex5.py for an implementation using legacy callbacks.

We are given a set of candidate locations J and a set of clients C.
Facilities should be opened in some of the candidate locations so that
the clients' demand can be served. The facility location problem consists
of deciding in which locations facilities should be opened and assigning
each client to a facility at a minimum cost.

A fixed cost is incurred when opening a facility, and a linear cost is
associated with the demand supplied from a given facility to a client.

Furthermore, each facility has a capacity and can only serve |C| - 1
clients. Note that in the variant of the problem considered here, each
client is served by only one facility.

The problem is formulated as a mixed integer linear program using binary
variables: used[j], for all locations j in J, indicating if a facility is
opened in location j; supply[c][j], for each client c in C and facility j
in J, indicating the demand of client c supplied from facility j.

Thus, the following model formulates the facility location problem:

 Minimize sum(j in J) fixedCost[j] * used[j] +
          sum(j in J) sum(c in C) cost[c][j] * supply[c][j]
 Subject to:
   sum(j in J) supply[c][j] == 1                    for all c in C,
   sum(c in C) supply[c][j] <= (|C| - 1) * used[j]  for all j in J,
   supply[c][j] in {0, 1}                   for all c in C, j in J,
   used[j] in {0, 1}                                for all j in J.

The first set of constraints are the demand constraints ensuring that the
demand of each client is satisfied. The second set of constraints are the
capacity constraints, ensuring that if a facility is placed in location j
the capacity of that facility is not exceeded.

The program in this file, formulates a facility location problem and
solves it. Furthermore, different cutting planes methods are implemented
using the generic callback API to help the solution of the problem:

- Disaggregated capacity cuts separated algorithmically (see the
  FacilityCallback.disaggregate method for details),

- Disaggregated capacity cuts separated using a cut table (see the
  FacilityCallback.cutsFromTable method for details),

- Capacity constraints separated as lazy constraints (see the
  FacilityCallback.lazyCapacity method for details).

Those different methods are invoked using the generic callback API.

See the usage message below for how to switch between these options.
"""
import sys
import traceback

import cplex
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


class FacilityCallback():
    """This is the class implementing the generic callback interface.

    It has three main functions:
       - disaggregate: add disaggregated constraints linking clients and
         location.
       - cuts_from_table: do the same using a cut table.
       - lazy_capacity: adds the capacity constraint as a lazy
         constraint.
    """

    def __init__(self, clients, locations, used, supply):
        self.clients = clients
        self.locations = locations
        self.used = used
        self.supply = supply
        self.cutlhs = None
        self.cutrhs = None

    def disaggregate(self, context):
        """Separate the disaggregated capacity constraints.

        In the model we have for each location j the constraint

        sum(c in clients) supply[c][j] <= (nbClients-1) * used[j]

        Clearly, a client can only be serviced from a location that is
        used, so we also have a constraint

        supply[c][j] <= used[j]

        that must be satisfied by every feasible solution. These
        constraints tend to be violated in LP relaxation. In this
        callback we separate them.
        """
        for j in self.locations:
            for c in self.clients:
                s, o = context.get_relaxation_point(
                    [self.supply[c][j], self.used[j]])
                if s > o + EPS:
                    print('Adding supply(%d)(%d) <= used(%d) [%f > %f]' %
                          (c, j, j, s, o))
                    cutmanagement = cplex.callbacks.UserCutCallback.use_cut.purge
                    context.add_user_cut(
                        cut=cplex.SparsePair([self.supply[c][j], self.used[j]],
                                             [1.0, -1.0]),
                        sense='L', rhs=0.0,
                        cutmanagement=cutmanagement,
                        local=False)

    def cuts_from_table(self, context):
        """Generate disaggregated constraints looking through a table.

        Variant of the disaggregate method that does not look for
        violated cuts dynamically.

        Instead it uses a static table of cuts and scans this table for
        violated cuts.
        """
        for lhs, rhs in zip(self.cutlhs, self.cutrhs):
            # Compute activity of left-hand side
            act = sum(c * x for c, x in zip(lhs.val,
                                            context.get_relaxation_point(lhs.ind)))
            if act > rhs + EPS:
                print('Adding %s [act = %f]' % (str(lhs), act))
                cutmanagement = cplex.callbacks.UserCutCallback.use_cut.purge
                context.add_user_cut(cut=lhs, sense="L", rhs=rhs,
                                     cutmanagement=cutmanagement, local=False)

    def lazy_capacity(self, context):
        """Lazy constraint callback to enforce the capacity constraints.

        If used then the callback is invoked for every integer feasible
        solution CPLEX finds. For each location j it checks whether
        constraint

           sum(c in C) supply[c][j] <= (|C| - 1) * opened[j]

        is satisfied. If not then it adds the violated constraint as lazy
        constraint.
        """
        # We only work with bounded models
        if not context.is_candidate_point():
            raise Exception('Unbounded solution')
        for j in self.locations:
            isused = context.get_candidate_point(self.used[j])
            served = sum(context.get_candidate_point(
                [self.supply[c][j] for c in self.clients]))
            if served > (len(self.clients) - 1.0) * isused + EPS:
                print('Adding lazy constraint %s <= %d*used(%d)' %
                      (' + '.join(['supply(%d)(%d)' % (x, j) for x in self.clients]),
                       len(self.clients) - 1, j))
                context.reject_candidate(
                    constraints=[cplex.SparsePair(
                        [self.supply[c][j]
                            for c in self.clients] + [self.used[j]],
                        [1.0] * len(self.clients) + [-(len(self.clients) - 1)]), ],
                    senses='L',
                    rhs=[0.0, ])

    def invoke(self, context):
        """Implements the required invoke method.

        This is the method that we have to implement to fulfill the
        generic callback contract. CPLEX will call this method during
        the solution process at the places that we asked for.
        """
        try:
            if context.in_relaxation():
                if self.cutlhs:
                    self.cuts_from_table(context)
                else:
                    self.disaggregate(context)
            elif context.in_candidate():
                self.lazy_capacity(context)
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise
#endif


def admipex8(datadir, from_table, lazy, use_callback):
    """Solve a capacitated facility location problem with cutting planes using
    the new generic callback interface.
    """
    # Read in data file. The data we read is
    # fixedcost  -- a list/array of facility fixed cost
    # cost       -- a matrix for the costs to serve each client by each
    #               facility

    # pylint: disable=unbalanced-tuple-unpacking
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
    cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair(supply[c], [1.0] * len(supply[c]))
                  for c in clients],
        senses=['E'] * len(clients),
        rhs=[1.0] * len(clients))

    # Capacity constraint for each location. We just require that a single
    # location cannot serve all clients, that is, the capacity of each
    # location is nbClients-1. This makes the model a little harder to
    # solve and allows us to separate more cuts.
    if not lazy:
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(
                [supply[c][j] for c in clients] + [used[j]],
                [1.0] * len(clients) + [-(len(clients) - 1.0)])
                for j in locations],
            senses=['L'] * len(locations),
            rhs=[0] * len(locations))

    # Tweak some CPLEX parameters so that CPLEX has a harder time to
    # solve the model and our cut separators can actually kick in.
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

    # Setup the callback.
    # We instantiate the callback object and attach the necessary data
    # to it.
    # We also setup the contexmask parameter to indicate when the callback
    # should be called.
    facilitycb = FacilityCallback(clients, locations, used, supply)
    contextmask = 0
    if use_callback:
        contextmask |= cplex.callbacks.Context.id.relaxation
        if from_table:
            # Generate all disaggregated constraints and put them into a
            # table that is scanned by the callback.
            facilitycb.cutlhs = [cplex.SparsePair([supply[c][j], used[j]],
                                                  [1.0, -1.0])
                                 for j in locations for c in clients]
            facilitycb.cutrhs = [0] * len(locations) * len(clients)
    if lazy:
        contextmask |= cplex.callbacks.Context.id.candidate

    # If contextMask is not zero we add the callback.
    if contextmask:
        cpx.set_callback(facilitycb, contextmask)

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



def main():
    """Set default arguments and parse command line."""
    # If a directory is not given on the command line, we use the
    # following default.
    datadir = '../../../examples/data'
    from_table = False
    lazy = False
    use_callback = True
    for arg in sys.argv[1:]:
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
            usage(sys.argv[0])
    admipex8(datadir, from_table, lazy, use_callback)


if __name__ == "__main__":
    main()
