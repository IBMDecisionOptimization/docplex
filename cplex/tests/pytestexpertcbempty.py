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
Simple tests with an empty expert callback:
invoke all query functions in all contexts
"""
import unittest
import random
import sys
import traceback

from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase
import cplex

RELDATADIR = '../../../examples/data'


class ExpertCallback():
    def __init__(self, obj, cname, rname):
        self.obj = obj
        self.cname = cname
        self.rname = rname
        self.calls = 0

    def _check_objval(self, x, objval):
        myobj = sum(x[j] * c for j, c in enumerate(self.obj))
        if abs(myobj - objval) > 1e-6:
            raise AssertionError('Unexpected objective value %f (expected %f)' %
                                 (myobj, objval))

    def _query_some(self, func, names):
        """Query some random items by name."""
        multi = set()
        for i in range(10):
            idx = random.randint(0, len(names) - 1)
            if (i % 2 == 0) and not names is None:
                func(names[idx])
                multi.add(names[idx])
            else:
                func(idx)
                multi.add(idx)
        func([j for j in multi])

    def invoke(self, context):
        try:
            self.calls += 1
            thread_id = context.get_int_info(
                cplex.callbacks.Context.info.thread_id)
            where = context.get_id()
            print('Callback invoked on thread %d in context %d (%d)' %
                  (thread_id, where, self.calls))
            query_restarts = False # Query number of restarts?
            query_aftercutloop = False # Can we query "is after cut loop"
            # Perform queries from callback. The idea is to cover (almost)
            # all functions in the API of class cplex.callbacks.Context
            if context.in_thread_up():
                context.get_long_info(cplex.callbacks.Context.info.threads)
            elif context.in_thread_down():
                context.get_long_info(cplex.callbacks.Context.info.threads)
            elif context.in_local_progress():
                context.get_long_info(cplex.callbacks.Context.info.node_count)
                context.get_double_info(cplex.callbacks.Context.info.time)
                context.get_double_info(
                    cplex.callbacks.Context.info.deterministic_time)
                context.get_int_info(cplex.callbacks.Context.info.feasible)
                context.get_double_info(
                    cplex.callbacks.Context.info.best_bound)
                context.get_double_info(
                    cplex.callbacks.Context.info.best_solution)
                query_restarts = True
                self._query_some(context.get_global_lower_bounds, self.cname)
                self._query_some(context.get_global_upper_bounds, self.cname)
            elif context.in_global_progress():
                context.get_long_info(cplex.callbacks.Context.info.node_count)
                context.get_double_info(cplex.callbacks.Context.info.time)
                context.get_double_info(
                    cplex.callbacks.Context.info.deterministic_time)
                context.get_double_info(
                    cplex.callbacks.Context.info.best_bound)
                context.get_double_info(
                    cplex.callbacks.Context.info.best_solution)
                query_restarts = True
                if context.get_int_info(cplex.callbacks.Context.info.feasible) != 0:
                    self._check_objval(context.get_incumbent(),
                                       context.get_incumbent_objective())
                    self._query_some(context.get_incumbent, self.cname)
                self._query_some(context.get_global_lower_bounds, self.cname)
                self._query_some(context.get_global_upper_bounds, self.cname)
            elif context.in_relaxation() or context.in_branching():
                self._check_objval(context.get_relaxation_point(),
                                   context.get_relaxation_objective())
                self._query_some(context.get_relaxation_point, self.cname)
                self._query_some(context.get_local_lower_bounds, self.cname)
                self._query_some(context.get_local_upper_bounds, self.cname)
                self._query_some(context.get_global_lower_bounds, self.cname)
                self._query_some(context.get_global_upper_bounds, self.cname)
                query_restarts = True
                if context.in_relaxation():
                    query_aftercutloop = True
            elif context.in_candidate():
                if not context.is_candidate_point():
                    raise AssertionError('Unbounded solution')
                self._check_objval(context.get_candidate_point(),
                                   context.get_candidate_objective())
                self._query_some(context.get_candidate_point, self.cname)
                self._query_some(context.get_global_lower_bounds, self.cname)
                self._query_some(context.get_global_upper_bounds, self.cname)
                query_restarts = True
                # Just make sure we can invoke that function. More elaborate
                # testing is in xtestexpertcb_solsource.c
                src = context.get_candidate_source()
                ok = False,
                if src == cplex.callbacks.IncumbentCallback.solution_source.node_solution:
                    ok = True
                if src == cplex.callbacks.IncumbentCallback.solution_source.user_solution:
                    ok = True
                if src == cplex.callbacks.IncumbentCallback.solution_source.heuristic_solution:
                    ok = True
                if src == cplex.callbacks.IncumbentCallback.solution_source.mipstart_solution:
                    ok = True
                if not ok:
                    raise AssertionError('Unexpected source %s' % str(src))

            if query_restarts:
                # Just make sure we can invoke that function. More elaborate
                # testing is at the C layer
                context.get_int_info(
                    cplex.callbacks.Context.info.restarts)
            if query_aftercutloop:
                # Just make sure we can invoke that function. More elaborate
                # testing is at the C layer
                context.get_int_info(
                    cplex.callbacks.Context.info.after_cut_loop)
 
            # Query unique id of current node.
            # This can only succeed in the Relaxation or Branching context
            # and must raise an exception in all other contexts.
            # For the candidate context it is allowed to succeed (for an
            # integral node) and fail (for a solution from a heuristic)
            try:
                context.get_long_info(cplex.callbacks.Context.info.node_uid)
                assert context.in_relaxation() or context.in_branching() or context.in_candidate()
            except CplexSolverError as cse:
                assert not (context.in_relaxation() or context.in_branching())
                assert cse.args[2] == error_codes.CPXERR_UNSUPPORTED_OPERATION

            # Query depth of current node.
            # This can only succeed in the Relaxation or Branching context
            # and may succeed in the candidate context. It must raise an
            # exception in all other contexts.
            try:
                context.get_long_info(cplex.callbacks.Context.info.node_depth)
                assert context.in_relaxation() or context.in_branching() or context.in_candidate()
            except CplexSolverError as cse:
                assert not (context.in_relaxation() or context.in_branching())
                assert cse.args[2] == error_codes.CPXERR_UNSUPPORTED_OPERATION

            # Query number of nodes left.
            # This must fail in ThreadUp and ThreadDown and must succeed in
            # Relaxation. In all other contexts it should usually succeed but
            # allowed to fail.
            try:
                context.get_long_info(cplex.callbacks.Context.info.nodes_left)
                assert not (context.in_thread_up() or context.in_thread_down())
            except CplexSolverError as cse:
                assert not context.in_relaxation()
                assert cse.args[2] == error_codes.CPXERR_UNSUPPORTED_OPERATION
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


class ExpertCallbackModifyTests(CplexTestCase):
    def _test_model(self, modelfile):
        with cplex.Cplex(modelfile) as cpx:
            cb = ExpertCallback(cpx.objective.get_linear(),
                                cpx.variables.get_names(),
                                cpx.linear_constraints.get_names())
            cpx.set_callback(cb, -1)
            cpx.solve()
            print(cb.calls, 'callback invocations')
            self.assertGreaterEqual(cb.calls, 1)

    def test_aflow30(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz')


if __name__ == '__main__':
    unittest.main()
