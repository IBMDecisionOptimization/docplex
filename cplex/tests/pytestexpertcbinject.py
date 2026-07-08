# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Simple tests with a callback that posts a heuristic solution.
"""
import unittest
import sys
import traceback

from cplextestcase import CplexTestCase
from cplex.callbacks import Context
import cplex
from cplex._internal._subinterfaces import SolutionStatus
from cplex.exceptions import error_codes
import testutil

RELDATADIR = '../../../examples/data'
EPSILON = 1e-6


class ExpertCallback():
    def __init__(self, sol, obj, soltype, status, thread, n, cname):
        self.sol = sol      # Solution to inject
        self.obj = obj      # Objective value for sol
        self.soltype = soltype  # Type with which to inject sol
        self.status = status   # Expected status of injection
        self.thread = thread   # Id of thread that should inject
        self.n = n        # Call at which thread should inject
        self.cname = cname    # Column names or None

    def _mapj(self, j):
        """Return either j or the name for variable j."""
        if self.cname is None:
            return j
        return self.cname[j] if (j % 2) == 0 else j

    def invoke(self, context):
        try:
            thread_no = context.get_int_info(Context.info.thread_id)
            if context.get_id() == Context.id.thread_up or \
                    context.get_id() == Context.id.thread_down:
                # Posting heuristic solutions in a thread_up or thread_down
                # is not supported
                try:
                    context.post_heuristic_solution(
                        cplex.SparsePair([self._mapj(j) for j in range(len(self.sol))],
                                         self.sol),
                        self.obj, self.soltype)
                    raise AssertionError('No exception!')
                except cplex.exceptions.CplexSolverError as e:
                    if e.args[2] != error_codes.CPXERR_UNSUPPORTED_OPERATION:
                        raise AssertionError('Wrong exception ' + str(e))
            elif thread_no == self.thread and self.status == 0 and self.n == 1:
                # This is the thread that must inject the solution and we expect
                # this to succeed
                self.n -= 1
                context.post_heuristic_solution(
                    cplex.SparsePair([self._mapj(j) for j in range(len(self.sol))],
                                     self.sol),
                    self.obj, self.soltype)
            elif thread_no == self.thread and self.status != 0 and self.n == 1:
                # This is the thread that must inject the solution but we expect
                # that to fail
                self.n -= 1
                try:
                    context.post_heuristic_solution(
                        cplex.SparsePair([self._mapj(j) for j in range(len(self.sol))],
                                         self.sol),
                        self.obj, self.soltype)
                    raise AssertionError('No exception!')
                except cplex.exceptions.CplexSolverError as e:
                    if e.args[2] != self.status:
                        raise AssertionError('Wrong status!' + str(e.args[2]) +
                                             ' vs ' + str(self.status))
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


class ExpertCallbackInjectTests(CplexTestCase):

    def _load(self, expand=False):
        """Load markshare1 and set parameters to that we stop after 1000
        nodes or once we find the optimal solution (without proving it)."""
        cpx = self._newCplex()
        testutil.create_markshare1(cpx)
        if expand:
            # Add an additional variable that is forced to 0 in any solution
            # This will make attempts to inject get_markshare1_optimal() as
            # complete solution fail.
            y = list(cpx.variables.add(lb=[0], ub=[1], names=['y']))[0]
            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair([0, y], [-1, 1]),
                                                 cplex.SparsePair([0, y], [1, 1])],
                                       senses=['L', 'L'],
                                       rhs=[0.0, 1.0])
        cpx.parameters.mip.limits.nodes.set(1000)
        cpx.parameters.mip.tolerances.absmipgap.set(1.5)
        return cpx

    def test_validate(self):
        """Make sure we don't find the optimal solution without injecting it."""
        with self._load() as cpx:
            cpx.solve()
            self.assertGreater(cpx.solution.get_objective_value(), 1.5)

    def _test_model(self, thread, count, expand, soltype, status):
        with self._load(expand) as cpx:
            cb = ExpertCallback(testutil.get_markshare1_optimal(), 1.0,
                                soltype, status, thread, count,
                                cpx.variables.get_names())
            cpx.set_callback(cb, -1)
            cpx.solve()
            return cpx.solution.get_objective_value(), cpx.solution.get_status()

    def test_inject_complete_thread0_call1(self):
        obj, status = self._test_model(0, 1, False,
                                       Context.solution_strategy.check_feasible,
                                       0)
        self.assertAlmostEqual(obj, 1.0, delta=1e-6)
        self.assertEqual(status, SolutionStatus.optimal_tolerance)

    def test_inject_complete_fail(self):
        obj, status = self._test_model(0, 1, True,
                                       Context.solution_strategy.check_feasible,
                                       error_codes.CPXERR_BAD_ARGUMENT)
        self.assertGreater(obj, 1.5)
        self.assertEqual(status, SolutionStatus.node_limit_feasible)

    def test_inject_presolved_thread0_call1(self):
        obj, status = self._test_model(0, 1, True,
                                       Context.solution_strategy.propagate,
                                       0)
        self.assertAlmostEqual(obj, 1.0, delta=1e-6)
        self.assertEqual(status, SolutionStatus.optimal_tolerance)


if __name__ == '__main__':
    unittest.main()
