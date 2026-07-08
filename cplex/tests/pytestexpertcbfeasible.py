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
Simple tests with a lazy constraint expert callback: Turn all linear
constraints into lazy constraints separated by the callback.
"""
import unittest
import sys
import traceback
from collections import namedtuple

from cplextestcase import CplexTestCase
from cplex.exceptions import CplexSolverError, error_codes
import cplex

RELDATADIR = '../../../examples/data'
EPSILON = 1e-6


class ExpertCallback():
    def __init__(self, cuts, accumulate, test, cname, reject_local):
        self.cuts = cuts
        self.cname = cname
        self.accumulate = accumulate
        self.qtype = 0
        self.test = test
        self.reject_local = reject_local
        # We only add a certain number of local lazy constraints.
        # After that number was added we add constraints as global constraints
        # so that the test runs faster.
        self.local_rejected = 0

    def _mapj(self, j):
        """Return either j or the name for variable j."""
        if self.cname is None:
            return j
        return self.cname[j] if (j % 2) == 0 else j

    def _getx(self, context, x):
        """Query values for x by index AND name and in the various forms possible"""
        if not context.is_candidate_point():
            raise AssertionError('Unbounded solution')
        # Get the type of query we want to perform.
        # NOTE: This is not deterministic since it depends on the order in which
        #       callbacks are executed. This should however not be a problem as
        #       far as the test and correctness are concerned.
        qtype = self.qtype % 4
        self.qtype += 1
        if qtype == 0:
            # Query the full vector
            vals = context.get_candidate_point()
            return [vals[j] for j in x]
        elif qtype == 1:
            # Query the values one by one individually
            return [context.get_candidate_point(self._mapj(j)) for j in x]
        elif qtype == 2:
            # Query the values by providing the list of variables
            return context.get_candidate_point([self._mapj(j) for j in x])
        elif qtype == 3:
            # Query the values by querying a range
            minj, maxj = min(x), max(x)
            vlist = context.get_candidate_point(self._mapj(minj),
                                                self._mapj(maxj))
            return [vlist[j - minj] for j in x]
        else:
            raise AssertionError('impossible')

    def invoke(self, context):
        try:
            inc = context.is_candidate_point()
            ray = context.is_candidate_ray()
            self.test.assertTrue(inc or ray)
            self.test.assertTrue(inc != ray)
            self.test.assertTrue(inc)  # True for models that we test so far

            do_local = self.reject_local and \
                       self.local_rejected < 1000 and \
                       context.get_candidate_source() \
                       == cplex.callbacks.IncumbentCallback.solution_source.node_solution

            if inc:
                try:
                    context.get_candidate_ray()
                except CplexSolverError as cse:
                    self.test.assertEqual(
                        cse.args[2], error_codes.CPXERR_CAND_NOT_RAY)
                violated = list()
                for c in self.cuts:
                    lhs = c.lhs
                    sense = c.sense
                    rhs = c.rhs
                    value = sum(a * x for a, x in zip(lhs.val,
                                                      self._getx(context, lhs.ind)))
                    if sense == 'E' and abs(value - rhs) > EPSILON:
                        if not self.accumulate:
                            if do_local:
                                context.reject_candidate_local(
                                    constraints=[lhs], senses=[sense],
                                    rhs=[rhs])
                                self.local_rejected += 1
                            else:
                                context.reject_candidate(
                                    constraints=[lhs], senses=[sense], rhs=[rhs])
                        violated.append([lhs, sense, rhs])
                    elif sense == 'L' and value > rhs + EPSILON:
                        if not self.accumulate:
                            if do_local:
                                context.reject_candidate_local(
                                    constraints=[lhs], senses=[sense],
                                    rhs=[rhs])
                                self.local_rejected += 1
                            else:
                                context.reject_candidate(
                                    constraints=[lhs], senses=[sense],
                                    rhs=[rhs])
                        violated.append([lhs, sense, rhs])
                    elif sense == 'G' and value < rhs - EPSILON:
                        if not self.accumulate:
                            if do_local:
                                context.reject_candidate_local(
                                    constraints=[lhs], senses=[sense],
                                    rhs=[rhs])
                                self.local_rejected += 1
                            else:
                                context.reject_candidate(
                                    constraints=[lhs], senses=[sense],
                                    rhs=[rhs])
                        violated.append([lhs, sense, rhs])
                if len(violated) > 0:
                    if self.accumulate:
                        if do_local:
                            context.reject_candidate_local(
                                constraints=[cplex.SparsePair(v[0].ind, v[0].val)
                                             for v in violated],
                                senses=[v[1]
                                        for v in violated],
                                rhs=[v[2] for v in violated])
                            self.local_rejected += 1
                        else:
                            context.reject_candidate(
                                constraints=[cplex.SparsePair(v[0].ind, v[0].val)
                                             for v in violated],
                                senses=[v[1]
                                        for v in violated],
                                rhs=[v[2] for v in violated])
                    print('Callback separated', len(violated), 'cuts')
            if ray:
                try:
                    context.get_candidate_point()
                except CplexSolverError as cse:
                    self.test.assertEqual(
                        cse.args[2], error_codes.CPXERR_CAND_NOT_POINT)
                try:
                    context.get_candidate_objective()
                except CplexSolverError as cse:
                    self.test.assertEqual(
                        cse.args[2], error_codes.CPXERR_CAND_NOT_POINT)
        except:
            info = sys.exc_info()
            print('#### Exception in callback: ', info[0])
            print('####                        ', info[1])
            print('####                        ', info[2])
            traceback.print_tb(info[2], file=sys.stdout)
            raise


class ExpertCallbackFeasibleTests(CplexTestCase):
    def _test_model(self, modelfile, obj, accumulate, usenames, reject_local):
        with cplex.Cplex(modelfile) as cpx:
            # Capture all linear constraints in the callback instance so that
            # it can separate them as lazy constraints.
            Constraint = namedtuple('Constraint', ['lhs', 'sense', 'rhs'])
            cb = ExpertCallback([Constraint(lhs=l, sense=s, rhs=r)
                                 for l, s, r in zip(cpx.linear_constraints.get_rows(),
                                                    cpx.linear_constraints.get_senses(),
                                                    cpx.linear_constraints.get_rhs())],
                                accumulate, self,
                                cpx.variables.get_names() if usenames else None,
                                reject_local)
            # Delete linear constraints.
            cpx.linear_constraints.delete(
                0, cpx.linear_constraints.get_num() - 1)
            # Solve the empty model to make sure it has an objective different
            # from the optimal one (so that the constraints actually make a
            # difference
            cpx.solve()
            self.assertNotAlmostEqual(cpx.solution.get_objective_value(), obj,
                                      delta=1e-6)
            # Now solve with callback and make sure we get the optimal solution
            cpx.parameters.advance.set(0)
            cpx.set_callback(cb, cplex.callbacks.Context.id.candidate)
            cpx.solve()
            self.assertAlmostEqual(cpx.solution.get_objective_value(), obj,
                                   delta=1e-6)
            self.assertGreaterEqual(cpx.solution.MIP.get_num_cuts(cpx.solution.MIP.cut_type.user),
                                    1)

    def test_aflow30(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz', 1158.0,
                         False, True, False)

    def test_aflow30_accumulate(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz', 1158.0,
                         True, True, False)

    def test_aflow30_local(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz', 1158.0,
                         False, True, True)

    def test_aflow30_accumulate_local(self):
        self._test_model(RELDATADIR + '/aflow30a.mps.gz', 1158.0,
                         True, True, True)


if __name__ == '__main__':
    unittest.main()
