# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2021
# --------------------------------------------------------------------------

"""
Problem Description
-------------------

This example illustrates how to use blackbox expressions to solve
a job-shop scheduling problem with uncertain operation durations.

It is an alternative approach to the stochastic optimization technique
illustrated in example sched_stochastic_jobshop.cpp.

The problem is an extension of the classical job-shop scheduling
problem (see sched_jobshop.cpp).

In the classical job-shop scheduling problem a finite set of jobs is
processed on a finite set of machines. Each job is characterized by a
fixed order of operations, each of which is to be processed on a
specific machine for a specified duration.  Each machine can process
at most one operation at a time and once an operation initiates
processing on a given machine it must complete processing
uninterrupted.  The objective of the problem is to find a schedule
that minimizes the makespan of the schedule.

In the present version of the problem, the duration of a given operation
is uncertain and supposed to be a uniform random variable varying in an
operation specific range [min, max]. The objective of the problem is to
find an ordering of the operations on the machines that minimizes
the average makespan of the schedule.

The estimation of the average makespan is computed by a blackbox expression
avg_makespan on the set of sequence variables of the model (see the use of
function mean_makespan). This blackbox function simulates the execution of the
specified sequences of operations on the machines on a number of samples and
returns an estimation of the average makespan.

The model uses this blackbox expression as objective function to be minimized.
Note that the model uses the average duration of operations (dmin+dmax)/2 as
deterministic duration to guide the search. One can show that the makespan
of the deterministic problem using the average durations is always lesser than
the average makespan, this is why it can be used as a lower bound on the blackbox
objective function.

The simulation simulates the execution of operations with uncertain durations under
precedence constraints in order to estimate the average makespan. Two techniques
are available to choose the durations: Monte-Carlo sampling and Descriptive Sampling
(depending on the definition of constant DESC_SAMPLING).

For each sample, Monte-Carlo sampling simply draws a random value in the
specified range for the duration of the each operation and simulates the execution
of this precedence graph to compute a sample of the makespan.

Descriptive sampling [1] is a more robust technique for sampling operations duration
in the context of simulating a precedence graph. It ensures that for a given operation,
the probability distibution of its duration is efficiently sampled by controling
the input set of sampled values.

Typically, a similar precision on the average makespan is achieved with 30 samples
of Descriptive sampling instead of typically 200 samples in Monte-Carlo sampling.

[1] E. Saliby. "Descriptive Sampling: A Better Approach to Monte Carlo Simulation".
The Journal of the Operational Research Society
Vol. 41, No. 12 (Dec., 1990), pp. 1133-1142

The blackbox expression avg_makespan is passed as data fields its scope
(the array of sequence variables representing the sequence of operations
on each machines), but other parameters for the durations and some housekeeping
structures are also passed using functools.partial.

For more information on blackbox expressions, see the concept "Blackbox expressions"
in the CP Optimizer C++ API Reference Manual.
"""

#
# Problem parameters
#

DEFAULT_FILENAME = "../../../examples/data/jobshop_blackbox_default.data"
DEFAULT_FILENAME = "data/jobshop_blackbox_default.data"
DEFAULT_TIMELIMIT = 30
DESC_SAMPLING = True
NUM_SAMPLES = 300 if DESC_SAMPLING else 2000
RANDOM_SEED = 1234

#
# Imports
#
import sys
try:
    import numpy as np
except ImportError:
    print("Please ensure you have installed module 'numpy'.")
    sys.exit(0)

try:
    import functools
except ImportError:
    print("Please ensure you have installed module 'functools'.")
    sys.exit(0)

try:
    import numba
except ImportError:
    print("Please ensure you have installed module 'numba'.")

from docplex.cp.utils import compare_natural
import docplex.cp.solver.solver as solver
# check Solver version
sol_version = solver.get_solver_version()
if compare_natural(sol_version, '22.1') < 0:
    print("Blackbox functions are not implemented in this solver version: {}".format(sol_version))
    print("This example cannot be run.")
    sys.exit(0)
    
from docplex.cp.model import CpoModel, CpoBlackboxFunction

# Core of the simulation.  The simulation has been pre-compiled into the
# 'cstream' variable (command stream) which is made up of a series of commands:
# <number-of-predecessors> [predecessors] <operation-id>
# This is used to build the makespan by taking the end time to be the duration
# of the operation plus the max end time of the predecessors.

@numba.jit(nopython=True)
def simulate(cstream, durations) :
    num_ops, num_samples = durations.shape
    end = np.empty(num_ops, dtype=np.float64)
    total_ms = 0
    for s in range(num_samples):
        cit = iter(cstream)
        for i in range(num_ops):
            n = next(cit)
            e = 0
            for k in range(n):
                e = max(e, end[next(cit)])
            o = next(cit)
            end[o] = e + durations[o, s]
        total_ms += np.max(end)
    rms = total_ms / num_samples
    return rms


# Three parts here.  First, we build the predecessor graph which
# means that for each operation we can associate the previous on
# the machine and previous in the job.  So, operations can have
# from 0 to 2 predecessors.
#
# Second, we order the operations topologically so that we can
# calculate the makespan in a purely feed-forward fashion to keep
# it fast.
#
# Thirdly, we perform the actual simulation.  Because we are doing
# this with numba, we will pass just the simplest of data structures
# to the simulation function as number sometimes has trouble with
# non-numeric data.

def mean_makespan(machine_sequences, # decided by CP Optimizer
                  itv_to_op,         # map: interval var -> operation index
                  durations          # durations per operation and scenario
):
    # They should be topologically ordered.
    num_ops = sum(len(ms) for ms in machine_sequences)

    # We assume a fixed job length = number of machines
    job_length = len(machine_sequences)

    # Build predecessors
    direct_pred = [None] * num_ops
    used = [0] * num_ops
    for ms in machine_sequences:
        prev = -1
        for r, itv in enumerate(ms):
            op_id = itv_to_op[itv]
            pred = ([op_id-1] if (op_id % job_length) != 0 else []) + ([prev] if r != 0 else [])
            for p in pred:
                used[p] += 1
            direct_pred[op_id] = pred
            prev = op_id

    # Build topological order
    active = [ i for i in range(num_ops) if used[i] == 0 ]
    order = []
    i = 0
    while i < len(active):
        a = active[i]
        order.append((a, direct_pred[a]))
        for p in direct_pred[a]:
            used[p] -= 1
            if used[p] == 0:
                active.append(p)
        i += 1
    order.reverse()

    # If we could not go over everything,
    # there is a loop in the precedence graph
    assert(len(active) == num_ops)

    # Simulate using numba.  So use simple data structures
    commands = []
    for i in range(num_ops):
        commands.append(len(order[i][1]))
        for p in order[i][1]:
            commands.append(p)
        commands.append(order[i][0])
    value = simulate(np.array(commands), durations)
    return value

#
# Make all durations for all operations in all scenarios
#
def make_durations(drange, num_samples, desc_sampling, seed):
    rng = np.random.default_rng(seed)
    num_ops = len(drange)
    durations = np.empty((num_ops, num_samples))
    for i in range(num_ops):
        dmin, dmax = drange[i]
        if desc_sampling:
            width = (dmax - dmin + 1) / num_samples
            for s in range(num_samples):
                durations[i,s] = np.floor(dmin + width * (s + rng.random()))
            rng.shuffle(durations[i,:])
        else:
            for s in range(num_samples):
                durations[i,s] = rng.integers(dmin, dmax + 1)
    return durations


def main(argv) :
    filename = DEFAULT_FILENAME
    time_limit = DEFAULT_TIMELIMIT
    if len(argv) > 1:
        if argv[1] != "-":
            filename = argv[1]
    if len(argv) > 2:
        time_limit = float(argv[2])

    try:
        with open(filename) as f:
            file_data = (int(elem) for elem in f.read().split())
    except FileNotFoundError as ex:
        print("Could not open {}".format(filename))
        print("Usage: {} <file> <time limit>".format(argv[0]))
        print("       Use '-' to mean the default input file")
        raise

    it = iter(file_data)
    num_jobs = next(it)
    num_machines = next(it)
    job_length = num_machines # for clarity
    itv_to_op = {}
    machine_ops = [ [] for _ in range(num_machines) ]
    mdl = CpoModel()
    op_id = 0
    drange = []
    job_ends = []
    for j in range(num_jobs):
        for op in range(job_length):
            machine = next(it)
            dmin = next(it)
            dmax = next(it)
            drange.append((dmin, dmax))
            itv = mdl.interval_var(length = (dmin + dmax)//2, name = "J_{}_{}".format(j, op))
            itv_to_op[itv] = op_id
            if op != 0:
                mdl.add(mdl.end_before_start(prev_itv, itv))
            machine_ops[machine].append(itv)
            op_id += 1
            prev_itv = itv
        job_ends.append(mdl.end_of(itv))
    classic_makespan = mdl.max(job_ends)

    # Sequences.  Needed for the back box evaluation
    machine_sequences = [ mdl.sequence_var(ops, name = "S_{}".format(i))
                          for i, ops in enumerate(machine_ops) ]
    # Only one job can execute at a time on a machine
    mdl.add(mdl.no_overlap(ms) for ms in machine_sequences)

    # Objective function based on black box
    durations = make_durations(drange, NUM_SAMPLES, DESC_SAMPLING, RANDOM_SEED)
    avg_makespan = CpoBlackboxFunction(functools.partial(mean_makespan,
                                                         itv_to_op = itv_to_op,
                                                         durations = durations))
    stochastic_makespan = avg_makespan(machine_sequences)
    mdl.add(stochastic_makespan >= classic_makespan)
    mdl.minimize(stochastic_makespan)

    result = mdl.solve(TimeLimit=time_limit, LogVerbosity="Normal", LogPeriod=10000, Workers=1)
    if result:
        print(result.get_objective_value())
    else:
        print("No solution found")


if __name__ == "__main__":
    main(sys.argv)
