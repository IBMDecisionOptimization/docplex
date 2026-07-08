# -*- coding: utf-8 -*-
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
Tests the Cplex object.

No command line arguments are required.
"""
import unittest
import errno
import multiprocessing  # New in Python 2.6
import os
import signal
import time

import cplex
from cplextestcase import CplexTestCase
from cplex.callbacks import MIPInfoCallback
from cplex.exceptions import CplexError, WrongNumberOfArgumentsError
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
import testutil

LP_EXAMPLE_FILE = '../../../examples/data/lpprog.lp'
MIP_EXAMPLE_FILE = '../../../examples/data/case1.lp'
POOL_EXAMPLE_FILE = '../../../examples/data/caso8.mps'
NOSWOT_EXAMPLE_FILE = '../../../examples/data/noswot.mps'

MY_SIGINT_VALUE = 42


def my_sigint_handler(signum, frame):
    raise KeyboardInterrupt(MY_SIGINT_VALUE)


class KeyboardInterruptCallback(MIPInfoCallback):
    """Simulate a Ctrl-C event every 1000 nodes."""

    def __init__(self, env):
        super().__init__(env)
        self.num_nodes = -1
        self.pid = -1
        self.numkilled = 0

    def __call__(self):
        nodes = self.get_num_nodes()
        if nodes > self.num_nodes + 1000:
            self.num_nodes = nodes
            self.numkilled += 1
            try:
                os.kill(self.pid, signal.SIGINT)
                os.waitpid(self.pid, 0)
            except OSError as e:
                # process may be finished by kill already
                if e.errno not in (errno.ECHILD, errno.ESRCH):
                    raise


class CplexTests(CplexTestCase):

    def testTooManyArgs(self):
        with self.assertRaises(WrongNumberOfArgumentsError):
            cplex.Cplex(1, 2, 3)

    def testInvalidArgument(self):
        with self.assertRaises(TypeError) as exc:
            cplex.Cplex(1)
        self.assertIn("invalid argument", str(exc.exception))

    def testSharedEnvironment(self):
        with self._newCplex() as cpx:
            with self.assertRaises(TypeError) as exc:
                cplex.Cplex(cpx, cpx._env)
            self.assertIn("invalid arguments", str(exc.exception))

    def testDeepCopy(self):
        cpx1 = self._newCplex()
        cpx1.read(MIP_EXAMPLE_FILE)
        cpx1.solve()
        self.assertEqual(cpx1.solution.get_status(),
                         cpx1.solution.status.MIP_optimal)
        self.assertEqual(cpx1.solution.get_objective_value(), 29.0)

        cpx2 = cplex.Cplex(cpx1)
        self._setAllStreams(cpx2, None)
        # TODO: There's no constant for the zero status ... Should we have one?
        self.assertEqual(cpx2.solution.get_status(), 0)
        try:
            self.assertEqual(cpx2.solution.get_objective_value(), 29.0)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_SOLN)
        cpx2.solve()
        self.assertEqual(cpx2.solution.get_status(),
                         cpx2.solution.status.MIP_optimal)
        self.assertEqual(cpx2.solution.get_objective_value(), 29.0)

    def testReadFileAndType(self):
        cpx = cplex.Cplex(MIP_EXAMPLE_FILE, 'lp')
        self.assertEqual(cpx.variables.get_num(), 4)
        # NB: We don't need to test with bad file type because testread
        # already exercises this.

    def testSetProblemType(self):
        cpx = self._newCplex()
        cpx.read("../../../examples/data/p0033.mps")
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.MILP)
        cpx.set_problem_type(cpx.problem_type.LP)
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.LP)
    
    def testSetProblemTypeOnSoln(self):
        cpx = self._newCplex()
        cpx.read("../../../examples/data/miqp0033.mps")
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.MIQP)
        cpx.set_problem_type(cpx.problem_type.MILP)
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.MILP)
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.MIP_optimal)
        self.assertTrue(cpx.solution.pool.get_num() > 0)
        cpx.set_problem_type(cpx.problem_type.fixed_MILP, 0)
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.fixed_MILP)

    def testIsMip(self):
        # NB: this is a "private" function.
        cpx = self._newCplex()
        cpx.read(MIP_EXAMPLE_FILE)
        self.assertTrue(cpx._is_MIP())

    def testIsNotMip(self):
        # NB: this is a "private" function.
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        self.assertFalse(cpx._is_MIP())

    # TODO: Can we test all of the "if" statements in _is_MIP()?

    # TODO: Test uncovered statements in solve()

    def testPopulateSolutionPool(self):
        cpx = self._newCplex()
        cpx.read(POOL_EXAMPLE_FILE)
        cpx.populate_solution_pool()
        # TODO: This is sort of slow ... is there a better model to use?
        # Default value of CPX_PARAM_POPULATELIM is 20.
        self.assertTrue(cpx.solution.pool.get_num() >= 20)

    def testCleanup(self):
        cpx = self._newCplex()
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.variables.add(obj=[1.0, 1.0, 1.0],
                          ub=[40.0, cplex.infinity, cplex.infinity])
        cpx.linear_constraints.add(
            lin_expr=[[[0, 1, 2], [0.0001, 1.0, 1.0]],
                      [[0, 1, 2], [1.0, -3.0, 1.0]]],
            senses="LL",
            rhs=[20.0, 30.0])
        row = cpx.variables.get_cols(0)
        self.assertEqual(row.ind, [0, 1])
        self.assertEqual(row.val, [0.0001, 1.0])
        cpx.cleanup(0.001)
        row = cpx.variables.get_cols(0)
        self.assertEqual(row.ind, [1])
        self.assertEqual(row.val, [1.0])

    def testNumCores(self):
        # Test by comparing with multiprocessing.cpu_count.  Should be
        # platform independent.
        cpx = self._newCplex()
        try:
            numcpu = multiprocessing.cpu_count()
            self.assertEqual(cpx.get_num_cores(), numcpu)
        except NotImplementedError:
            # If multiprocessing.cpu_count is not implemented on this
            # platform, then skip this test.
            pass

    def testGetTime(self):
        cpx = self._newCplex()
        start_time = cpx.get_time()
        sleep_time = 1  # in seconds
        time.sleep(sleep_time)
        end_time = cpx.get_time()
        self.assertTrue(end_time > start_time)
        time_elapsed = end_time - start_time
        self.assertTrue(
            time_elapsed * 1.01 >= sleep_time,
            "time_elapsed: {0}, sleep_time: {1}".format(
                time_elapsed, sleep_time))

    def testGetDetTime(self):
        cpx = self._newCplex()
        cpx.read(MIP_EXAMPLE_FILE)
        start_time = cpx.get_dettime()
        cpx.solve()
        end_time = cpx.get_dettime()
        self.assertTrue(end_time > start_time)

    def testCtrlCHandler(self):
        """Install a SIGINT handler and make sure it works after a solve.
        """
        # Signal handling does not work on Windows when run via cygwin,
        # so we skip this test there.
        if self.iswindows():
            return

        pid = os.getpid()
        signal.signal(signal.SIGINT, my_sigint_handler)

        def check_sigint_handler():
            self.assertEqual(signal.getsignal(signal.SIGINT),
                             my_sigint_handler)
            try:
                os.kill(pid, signal.SIGINT)
                self.fail()
            except KeyboardInterrupt as err:
                self.assertEqual(err.args[0], MY_SIGINT_VALUE)

        check_sigint_handler()
        with self._newCplex() as cpx:
            cpx.solve()
        check_sigint_handler()

    def testCtrlCHandlerInLoop(self):
        """Make sure Ctrl-C handling works in a solve loop."""
        # Signal handling does not work on Windows when run via cygwin,
        # so we skip this test there.
        if self.iswindows():
            return

        with self._newCplex() as cpx:
            cpx.read(self._getResource("examples/data/noswot.mps"))
            cb = cpx.register_callback(KeyboardInterruptCallback)
            cb.pid = os.getpid()
            num_nodes = -1
            for i in range(3):
                cpx.solve()
                status = cpx.solution.get_status()
                self.assertEqual(
                    cpx.solution.status.MIP_abort_feasible, status,
                    "Failed on solve #{0}, status {1}".format(i, status))
                self.assertGreater(
                    cb.num_nodes, num_nodes,
                    "Failed on solve #{0}, ! {1} > {2}".format(
                        i, cb.num_nodes, num_nodes))
                num_nodes = cb.num_nodes

    def testGetIndicesNotImplemented(self):
        """Sub-interfaces that are not indexed cannot use get_indices."""
        with self._newCplex() as cpx:
            ifacelist = [cpx.objective,
                         cpx.solution,
                         cpx.presolve,
                         cpx.feasopt,
                         cpx.conflict,
                         cpx.advanced,
                         cpx.order,
                         cpx.start]
            for iface in ifacelist:
                try:
                    iface.get_indices('bogus')
                    self.fail()
                except NotImplementedError:
                    pass

    def get_lpex1(self, set_types=False):
        cpx = self._newCplex()
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        varind = list(cpx.variables.add(
            obj=[1.0, 2.0, 3.0],
            ub=[40.0, cplex.infinity, cplex.infinity]
        ))
        if set_types:
            cpx.variables.set_types([(i, 'C') for i in varind])
        cpx.linear_constraints.add(
            lin_expr=[[varind, [-1.0, 1.0, 1.0]],
                      [varind, [1.0, -3.0, 1.0]]],
            senses="LL",
            rhs=[20.0, 30.0]
        )
        return cpx

    def testExplicitContinuousLP(self):
        """If ctype != NULL, then we should end up calling CPXmipopt."""
        cpx = self.get_lpex1(set_types=True)
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.MILP)
        cpx.solve()
        self.assertEqual(cpx.solution.get_method(), cpx.solution.method.MIP)

    def testImplicitContinuousLP(self):
        """If ctype == NULL, then we should end up calling CPXlpopt."""
        cpx = self.get_lpex1(set_types=False)
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.LP)
        cpx.solve()
        self.assertNotEqual(cpx.solution.get_method(), cpx.solution.method.MIP)

    def get_qpex1(self, set_types=False):
        cpx = self._newCplex()
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        varind = list(cpx.variables.add(
            obj=[1.0, 2.0, 3.0],
            ub=[40.0, cplex.infinity, cplex.infinity]
        ))
        if set_types:
            cpx.variables.set_types([(i, 'C') for i in varind])
        cpx.linear_constraints.add(
            lin_expr=[[varind, [-1.0, 1.0, 1.0]],
                      [varind, [1.0, -3.0, 1.0]]],
            senses="LL",
            rhs=[20.0, 30.0]
        )
        cpx.objective.set_quadratic([[[0, 1], [-33.0, 6.0]],
                                     [[0, 1, 2], [6.0, -22.0, 11.5]],
                                     [[1, 2], [11.5, -11.0]]])
        return cpx

    def testExplicitContinuousQP(self):
        """If ctype != NULL, then we should end up calling CPXmipopt."""
        cpx = self.get_qpex1(set_types=True)
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.MIQP)
        cpx.solve()
        self.assertEqual(cpx.solution.get_method(), cpx.solution.method.MIP)

    def testImplicitContinuousQP(self):
        """If ctype == NULL, then we should end up calling CPXqpopt."""
        cpx = self.get_qpex1(set_types=False)
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.QP)
        cpx.solve()
        self.assertNotEqual(cpx.solution.get_method(), cpx.solution.method.MIP)

    def testAll(self):
        """Test that __all__ contains only names that are actually exported."""

        missing = set(n for n in cplex.__all__
                      if getattr(cplex, n, None) is None)
        self.assertFalse(missing,
                         msg="__all__ contains unresolved names: %s" % (
                             ", ".join(missing),))

    def testClone(self):
        with self._newCplex(MIP_EXAMPLE_FILE) as orig, \
             self._newCplex(orig) as copy:
            # First, compare stats.
            orig_stats = orig.get_stats()
            copy_stats = orig.get_stats()
            for attr, orig_value in [(a, v)
                                     for a, v in orig_stats.__dict__.items()
                                     if not a.startswith("_")]:
                # By not specifying default for getattr, we ensure that
                # every attribute exists (otherwise, we would get an
                # AttributeError).
                copy_value = getattr(copy_stats, attr)
                self.assertEqual(orig_value, copy_value)
            self.assertEqual(str(orig_stats), str(copy_stats))
            # Second, as an extra sanity check, compare some basic stats
            # manually.
            self.assertEqual(orig.variables.get_num(),
                             copy.variables.get_num())
            self.assertEqual(orig.linear_constraints.get_num(),
                             copy.linear_constraints.get_num())

    def testQcpWithAll(self):
        """Add coverage for QCP with all possible LP methods."""
        with self._newCplex() as cpx:
            testutil.create_socpex1(cpx)
            cpx.parameters.advance.set(cpx.parameters.advance.values.none)
            for alg in cpx.parameters.lpmethod.values:
                cpx.parameters.lpmethod.set(alg)
                cpx.solve()
                self.assertEqual(cpx.solution.status.optimal,
                                 cpx.solution.get_status())
                self.assertAlmostEqual(10.0, cpx.solution.get_objective_value())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
