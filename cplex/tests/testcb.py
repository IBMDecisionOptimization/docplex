# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Tests callbacks.

No command line arguments are required.
"""
import unittest
import cplex
from cplex import SparsePair
from cplex.callbacks import (BranchCallback,
                             ContinuousCallback,
                             HeuristicCallback,
                             MIPCallback,
                             SolveCallback,
                             TuningCallback,
                             UserCutCallback,
                             MIPInfoCallback)
from cplex.exceptions import CplexError
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
import cplex._internal._constants as _constants
from cplextestcase import CplexTestCase
import testutil

BRANCH_CALLBACK_EXAMPLE = "../../data/caso8.mps"
USER_CUT_CALLBACK_EXAMPLE = "../../data/noswot.mps"


class MyContinuousCallback(ContinuousCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False
        self.num_cols = 0
        self.num_rows = 0
        self.num_qconstrs = 0
        self.force_error = False

    def __call__(self):
        if not self.called_once:
            self.called_once = True
            self.num_cols = self.get_num_cols()
            self.num_rows = self.get_num_rows()
            self.num_qconstrs = self.get_num_quadratic_constraints()
            if self.force_error:
                # Expecting this to raise a CplexSolverError and result in a
                # call to SWIG_callback.cb_geterrorstring.  This, for code
                # coverage.
                self._get_col_index("bogus_name")


class MySolveCallback(SolveCallback):

    def __init__(self, env):
        super().__init__(env)
        self.times_called = 0
        self.dual_feasible = False
        self.primal_feasible = False

    def __call__(self):
        self.times_called += 1
        if not self.dual_feasible or not self.primal_feasible:
            self.solve()
            self.dual_feasible = self.is_dual_feasible()
            self.primal_feasible = self.is_primal_feasible()


class AnotherSolveCallback(SolveCallback):

    def __init__(self, env):
        super().__init__(env)
        self.solved_once = False
        self.solve_alg = self.method.dual

    def __call__(self):
        if not self.solved_once:
            self.solved_once = self.solve(alg=self.solve_alg)


class SetStartCallback(SolveCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False

    def __call__(self):
        if not self.called_once:
            self.called_once = True
            pidx = list(range(self.get_num_cols()))
            didx = list(range(self.get_num_rows()))
            prim = [0.0] * self.get_num_cols()
            dual = [0.0] * self.get_num_rows()
            self.set_start(primal=SparsePair(pidx, prim),
                           dual=SparsePair(didx, dual))
            self.solve()


class AborterCallback(SolveCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False
        self.aborter = None

    def __call__(self):
        if not self.called_once:
            self.called_once = True
            self.aborter.abort()


class AbortCallback(SolveCallback):
    def __init__(self, env):
        super().__init__(env)
        self.called_once = False

    def __call__(self):
        if not self.called_once:
            self.called_once = True
            self.abort()


class MyHeuristicCallback(HeuristicCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False
        self.ub_before = 0
        self.ub_after = 0

    def __call__(self):
        if not self.called_once:
            self.called_once = True
            lbs = self.get_lower_bounds()
            ubs = self.get_upper_bounds()
            for i, v in enumerate(ubs):
                self.ub_before = v
                if v == cplex.infinity:
                    new_ub = 1000
                else:
                    new_ub = ubs[i] + 1
                try:
                    self.set_bounds(i, lbs[i], new_ub)
                    self.solve()
                    break
                except CplexError:
                    # If we run into a "Variable removed by presolve: cannot
                    # change bounds" then, try the next one.
                    pass
            else:
                raise AssertionError("Was not able to call set_bounds!")
            self.ub_after = self.get_upper_bounds(i)
            self.abort()


class MyTuningCallback(TuningCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False
        self.progress = 0.0

    def __call__(self):
        if not self.called_once:
            self.called_once = True
            self.progress = self.get_progress()


class MyMIPCallback(MIPCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False
        self.objcoefs = []

    def __call__(self):
        if not self.called_once:
            self.called_once = True
            self.objcoefs = self.get_objective_coefficients()
            self.abort()


class MyBranchCallback(BranchCallback):

    def __init__(self, env):
        super().__init__(env)
        self.called_once = False
        self.numbranches = 0
        self.btype = None
        self.node_id = 0
        self.integer_feasible = False

    def __call__(self):
        if self.called_once:
            # Only abort after this has been called once so that we allow for
            # at least one branch to be created.
            self.abort()
        else:
            self.called_once = True
            self.numbranches = self.get_num_branches()
            self.btype = self.get_branch_type()
            self.node_id = self.get_node_ID()
            self.integer_feasible = self.is_integer_feasible()


class Rtc19467Callback(UserCutCallback):

    def __init__(self, env):
        super().__init__(env)
        self.nodes = set()

    def __call__(self):
        node_id = self.get_node_ID()
        self.nodes.add(node_id)
        if len(self.nodes) == 10:
            self.abort()


class CallbackTestsWithoutPresolve(CplexTestCase):

    def testContinuousCallbackOnEmpty(self):
        with self._newCplex() as c:
            cb = c.register_callback(MyContinuousCallback)
            self.assertFalse(cb.called_once)
            self.assertEqual(cb.num_cols, 0)
            self.assertEqual(cb.num_rows, 0)
            self.assertEqual(cb.num_qconstrs, 0)
            self.solve(c)
            self.assertFalse(cb.called_once)
            self.assertEqual(cb.num_cols, 0)
            self.assertEqual(cb.num_rows, 0)
            self.assertEqual(cb.num_qconstrs, 0)

    def testContinuousCallbackOnSimple(self):
        with self._newCplex() as c:
            self._buildLpModel(c)
            cb = c.register_callback(MyContinuousCallback)
            self.assertFalse(cb.called_once)
            self.assertEqual(cb.num_cols, 0)
            self.assertEqual(cb.num_rows, 0)
            self.assertEqual(cb.num_qconstrs, 0)
            self.solve(c)
            self.assertEqual(c.solution.get_status_string(), "optimal")
            self.assertEqual(c.solution.get_objective_value(), 202.5)
            self.assertTrue(cb.called_once)
            self.assertTrue(cb.num_cols)
            self.assertTrue(cb.num_rows)
            self.assertEqual(cb.num_cols, c.variables.get_num())
            self.assertEqual(cb.num_rows, c.linear_constraints.get_num())
            self.assertEqual(cb.num_qconstrs, 0)

    def testGetErrorStringFromCallback(self):
        with self._newCplex() as c:
            self._buildLpModel(c)
            cb = c.register_callback(MyContinuousCallback)
            cb.force_error = True
            with self.assertRaises(CplexSolverError) as err:
                self.solve(c)
            self.assertEqual(err.exception.args[2],
                             error_codes.CPXERR_NO_NAMES)
            self.assertTrue(err.exception.args[0],
                            "Expecting non-empty error string!")

    def testGetNumQuadraticConstraints(self):
        with self._newCplex() as c:
            self._buildQcpModel(c)
            cb = c.register_callback(MyContinuousCallback)
            self.assertFalse(cb.called_once)
            self.assertEqual(cb.num_cols, 0)
            self.assertEqual(cb.num_rows, 0)
            self.assertEqual(cb.num_qconstrs, 0)
            self.solve(c)
            self.assertEqual(c.solution.get_status_string(), "optimal")
            self.assertAlmostEqual(c.solution.get_objective_value(),
                                   2.00234655688731, places=6)
            self.assertTrue(cb.called_once)
            self.assertTrue(cb.num_cols)
            self.assertTrue(cb.num_rows)
            self.assertTrue(cb.num_qconstrs)
            self.assertEqual(cb.num_cols, c.variables.get_num())
            self.assertEqual(cb.num_rows, c.linear_constraints.get_num())
            self.assertEqual(cb.num_qconstrs, c.quadratic_constraints.get_num())

    def testIsDualPrimalFeasible(self):
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(MySolveCallback)
            self.assertEqual(cb.times_called, 0)
            self.assertFalse(cb.dual_feasible)
            self.assertFalse(cb.primal_feasible)
            self.solve(c)
            self.assertEqual(c.solution.get_status_string(),
                             "integer optimal solution")
            self.assertEqual(c.solution.get_objective_value(), 122.5)
            self.assertTrue(cb.times_called)
            self.assertTrue(c.solution.is_dual_feasible())
            self.assertTrue(c.solution.is_primal_feasible())
            self.assertTrue(cb.dual_feasible)
            self.assertTrue(cb.primal_feasible)

    def testSubProbBarOpt(self):
        """Trigger call to cb_hybbaropt in SWIG_callback.c."""
        self._testSolveWithMethod(SolveCallback.method.barrier)

    def testSubProbNetOpt(self):
        """Trigger call to cb_hybnetopt in SWIG_callback.c."""
        self._testSolveWithMethod(SolveCallback.method.network)

    def testSetBounds(self):
        """Triggers calls to cb_getprestat_c and cb_chgbnds."""
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(MyHeuristicCallback)
            self.assertFalse(cb.called_once)
            self.assertEqual(cb.ub_before, 0)
            self.assertEqual(cb.ub_after, 0)
            self.solve(c)
            self.assertTrue(cb.called_once)
            # The HeuristicCallback.set_bounds() method only modifies
            # bounds locally (for the scope of the callback invocation).
            # The bounds are automatically reset when the callback
            # returns. The get_upper_bounds method does not return the
            # modified bounds.
            self.assertEqual(cb.ub_before, cb.ub_after)

    def testSetStart(self):
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(SetStartCallback)
            self.assertFalse(cb.called_once)
            self.solve(c)
            self.assertTrue(cb.called_once)

    def testAborterFromCallback(self):
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(AborterCallback)
            cb.aborter = c.use_aborter(cplex.Aborter())
            self.assertFalse(cb.called_once)
            self.solve(c)
            self.assertTrue(cb.called_once)
            self.assertEqual(c.solution.get_status(),
                             _constants.CPXMIP_ABORT_FEAS)

    def testAbortFromCallback(self):
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(AbortCallback)
            self.assertFalse(cb.called_once)
            self.solve(c)
            self.assertTrue(cb.called_once)
            self.assertEqual(c.solution.get_status(), _constants.CPXMIP_ABORT_FEAS)

    def testTuningCallback(self):
        with self._newCplex() as c:
            self._buildLpModel(c)
            cb = c.register_callback(MyTuningCallback)
            self.assertFalse(cb.called_once)
            self.assertEqual(cb.progress, 0.0)
            status = c.parameters.tune_problem()
            self.assertEqual(c.parameters.tuning_status[status], 'completed')
            self.assertTrue(cb.called_once)
            self.assertTrue(cb.progress)

    def testMipCallback(self):
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(MyMIPCallback)
            self.assertFalse(cb.called_once)
            self.assertFalse(cb.objcoefs)
            self.solve(c)
            self.assertTrue(cb.called_once)
            self.assertTrue(cb.objcoefs)

    def testBranchCallback(self):
        with self._newCplex() as c:
            c.read(BRANCH_CALLBACK_EXAMPLE)
            cb = c.register_callback(MyBranchCallback)
            self.assertFalse(cb.called_once)
            self.assertFalse(cb.numbranches)
            self.assertFalse(cb.btype)
            self.assertFalse(cb.node_id)
            self.assertFalse(cb.integer_feasible)
            self.solve(c)
            self.assertTrue(cb.called_once)
            self.assertEqual(cb.integer_feasible, cb.numbranches == 0)
            self.assertTrue(cb.btype in (cb.branch_type.any,
                                         cb.branch_type.SOS1,
                                         cb.branch_type.SOS2,
                                         cb.branch_type.variable))
            self.assertEqual(0, cb.node_id, "first node should have ID 0")

    def testRtc19467(self):
        with self._newCplex() as c:
            c.read(USER_CUT_CALLBACK_EXAMPLE)
            # Need to use traditional branch-and-cut to allow for control
            # callbacks.
            c.parameters.mip.strategy.search.set(
                c.parameters.mip.strategy.search.values.traditional)
            cb = c.register_callback(Rtc19467Callback)
            self.assertEqual(len(cb.nodes), 0)
            self.solve(c)
            self.assertEqual(len(cb.nodes), 10)

    def solve(self, cpx):
        cpx.parameters.preprocessing.presolve.set(False)
        cpx.solve()

    def _testSolveWithMethod(self, method):
        """Solve subproblem with specific method."""
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(AnotherSolveCallback)
            cb.solve_alg = method
            self.assertFalse(cb.solved_once)
            self.assertEqual(cb.solve_alg, method)
            self.solve(c)
            self.assertTrue(cb.solved_once)

    def _buildQcpModel(self, cpx):
        """Builds the simple model from qcpex1.py."""
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.linear_constraints.add(rhs=[20., 30.], senses="LL")
        cpx.variables.add(obj=[1., 2., 3.],
                          ub=[40., cplex.infinity, cplex.infinity],
                          columns=[[[0,1],[-1.0, 1.0]],
                                   [[0,1],[ 1.0,-3.0]],
                                   [[0,1],[ 1.0, 1.0]]])
        cpx.objective.set_quadratic([[[0,1],[-33.0, 6.0]],
                                     [[0,1,2],[ 6.0,-22.0, 11.5]],
                                     [[1,2],[ 11.5, -11.0]]])
        Q = cplex.SparseTriple(ind1=[0, 1, 2],
                               ind2=[0, 1, 2],
                               val=[1.0] * 3)
        cpx.quadratic_constraints.add(rhs=1., quad_expr=Q)

    def _buildMipModel(self, cpx):
        """Builds the MIP model from mipex1.py."""
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.variables.add(obj=[1., 2., 3., 1.],
                          lb=[0., 0., 0., 2.],
                          ub=[40., cplex.infinity, cplex.infinity, 3.],
                          types="CCCI")
        cpx.linear_constraints.add(lin_expr=[[[0, 1, 2, 3],
                                              [-1., 1., 1., 10.]],
                                             [[0, 1, 2],
                                              [1., -3., 1.]],
                                             [[1, 3], [1., -3.5]]],
                                   senses="LLE",
                                   rhs=[20., 30., 0.])

    def _buildLpModel(self, cpx):
        """Builds the simple model from lpex1.py."""
        testutil.create_lpex1(cpx)
        cpx.advanced.delete_names()


class CallbackTestsWithPresolve(CallbackTestsWithoutPresolve):

    def testSetStart(self):
        with self._newCplex() as c:
            self._buildMipModel(c)
            cb = c.register_callback(SetStartCallback)
            self.assertFalse(cb.called_once)
            with self.assertRaises(CplexError) as err:
                self.solve(c)
            self.assertEqual(str(err.exception), "Presolve must be disabled "
                             "to set dual vectors in SolveCallback.set_start")

    def solve(self, cpx):
        cpx.parameters.preprocessing.presolve.set(True)
        cpx.solve()


class InvalidCutType(MIPInfoCallback):

    def __call__(self):
        invalid_cut_type = 99
        count = self.get_num_cuts(invalid_cut_type)


class TestInvalidCutType(CplexTestCase):

    def testInvalidCutType(self):
        with self._newCplex() as c:
            c.read(self._getResource("examples/data/noswot.mps"))
            c.register_callback(InvalidCutType)
            with self.assertRaises(ValueError):
                c.solve()


class InvalidQualityMetricType(MIPInfoCallback):

    def __call__(self):
        invalid_value = 99
        float_quality = self.get_float_quality(invalid_value)


class TestInvalidQualityMetricType(CplexTestCase):

    def testInvalidQualityMetricType(self):
        with self._newCplex() as c:
            c.read(self._getResource("examples/data/noswot.mps"))
            c.register_callback(InvalidQualityMetricType)
            with self.assertRaises(ValueError):
                c.solve()


class BogusCallback():
    """Does not inherit from one of the legacy callback classes."""

    def __init__(self, env):
        pass


class BogusCallbackTests(CplexTestCase):

    def testBogusCallback(self):
        with self._newCplex() as cpx:
            with self.assertRaises(CplexError):
                cpx.register_callback(BogusCallback)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
