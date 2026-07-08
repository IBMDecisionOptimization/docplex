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
Tests tuning methods.

No command line arguments are required.
"""
import unittest
import re

from cplextestcase import CplexTestCase
import cplex
from cplex.exceptions import CplexError, CplexSolverError, error_codes
from cplex import SparsePair, SparseTriple, Aborter


class RepeatProcessor():
    """File-like object that processes CPLEX tuning output."""

    def __init__(self):
        self.count = 0

    def write(self, line):
        if re.search("Tuning on problem", line):
            self.count += 1

    def flush(self):
        pass


class BogusParameter():
    """A bogus parameter class used for error checking.

    This is only used to exercise some validation code.
    """
    pass


class SharedTuningTests():
    """Shared tests for tune_problem and tune_problem_set.

    As this class does not inherit from unittest.TestCase it won't
    execute by itself.
    """

    def tune(self, cpx, paramset=None):
        raise NotImplementedError

    def testSimple(self):
        cpx = self._newCplex()
        status = self.tune(cpx)
        self.assertEqual(status, cpx.parameters.tuning_status.completed)

    def testWithFixedParams(self):
        cpx = self._newCplex()
        status = self.tune(
            cpx, [(cpx.parameters.preprocessing.reduce,
                   cpx.parameters.preprocessing.reduce.values.none)])
        self.assertEqual(status, cpx.parameters.tuning_status.completed)

    def testWithFixedParamSet(self):
        cpx = self._newCplex()
        ps = cpx.create_parameter_set()
        ps.add(cpx.parameters.preprocessing.reduce,
               cpx.parameters.preprocessing.reduce.values.none)
        status = self.tune(cpx, ps)
        self.assertEqual(status, cpx.parameters.tuning_status.completed)

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testWithBadParamSet(self):
        cpx1 = self._newCplex()
        cpx2 = self._newCplex()
        ps2 = cpx2.create_parameter_set()
        self.assertRaises(ValueError, self.tune, cpx1, ps2)

    def testNoSolutionAfterTune(self):
        """After tuning the problem object should not have a solution."""
        cpx = self._newCplex()
        status = self.tune(cpx)
        self.assertEqual(status, cpx.parameters.tuning_status.completed)
        self.assertEqual("Unknown status value",
                         cpx.solution.get_status_string())
        self.assertEqual(cpx.solution.status.unknown,
                         cpx.solution.get_status())

    def testParamsBeforeAfterTune(self):
        """Test combined fixed and tuned parameters.

        After CPXXtuneparam/CPXtuneparam has finished, the
        environment should contain the combined fixed and tuned
        settings which the user can query or write to a file.
        """
        cpx = self._newCplex()
        cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.dual)
        changed_before = cpx.parameters.get_changed()
        # Other parameters may have been set via ILOG_CPLEX_PARAMETER_FILE
        # (see testallparams.py:testChangedParams). So, we don't check
        # for a strict number here.
        self.assertGreater(len(changed_before), 0)
        status = self.tune(
            cpx, [(cpx.parameters.preprocessing.reduce,
                   cpx.parameters.preprocessing.reduce.values.none)])
        self.assertEqual(status, cpx.parameters.tuning_status.completed)
        changed_after = cpx.parameters.get_changed()
        # In this case, we expect that the parameters that were set
        # before the call to tune are lost and only the fixed parameter
        # remains.
        self.assertEqual(len(changed_after), 1, changed_after)
        param, value = changed_after[0]
        self.assertEqual(param, cpx.parameters.preprocessing.reduce)
        self.assertEqual(
            value, cpx.parameters.preprocessing.reduce.values.none)

    def testWithDuplicateParams(self):
        # NOTE: If we remove the validation code tested here, we don't
        # get any type of error from the Callable Library.
        cpx = self._newCplex()
        try:
            self.tune(
                cpx, [(cpx.parameters.lpmethod,
                       cpx.parameters.lpmethod.values.dual),
                      (cpx.parameters.lpmethod,
                       cpx.parameters.lpmethod.values.dual)])
            self.assertFalse(__debug__)
        except CplexError as err:
            self.assertIn("duplicate parameters detected", str(err))

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testInvalidParamArgumentDebug(self):
        self.checkBogusParamArgumentDebug(42)

    @unittest.skipUnless(__debug__, "only test with non-optimized bytecode")
    def testBogusParamArgumentDebug(self):
        self.checkBogusParamArgumentDebug(BogusParameter())

    def checkBogusParamArgumentDebug(self, param):
        # NOTE: If we remove the validation code tested here, we don't
        # get any type of error from the Callable Library.
        try:
            self.checkWithBogusParam(BogusParameter())
            self.fail()
        except TypeError as err:
            self.assertIn(
                "invalid fixed_parameters_and_values arg detected", str(err))

    @unittest.skipIf(__debug__, "only test with optimized bytecode")
    def testInvalidParamArgumentOptimized(self):
        self.checkBogusParamArgumentOptimized(42)

    @unittest.skipIf(__debug__, "only test with optimized bytecode")
    def testBogusParamArgumentOptimized(self):
        self.checkBogusParamArgumentOptimized(BogusParameter())

    def checkBogusParamArgumentOptimized(self, param):
        try:
            self.checkWithBogusParam(param)
            self.fail()
        except AttributeError:
            pass

    def checkWithBogusParam(self, param):
        cpx = self._newCplex()
        self.tune(cpx, [(param, 0)])

    def testWithQCP(self):
        """Test incompatible problem type.

        In the documentation it says that "This routine does not apply
        to network models, nor to quadratically constrained programming
        problems (QCP)." We expect tuning to be a no-op here.
        """
        raise NotImplementedError

    def testWithZeroTimeLimit(self):
        # Skip this test on Windows (see RTC-34389).
        if self.iswindows():
            return
        cpx = self._newCplex()
        cpx.parameters.timelimit.set(0)
        status = self.tune(cpx)
        self.assertEqual(status, cpx.parameters.tuning_status.time_limit)

    def testWithZeroTuneTimeLimit(self):
        cpx = self._newCplex()
        cpx.parameters.tune.timelimit.set(0)
        status = self.tune(cpx)
        # CPX_PARAM_TUNINGTILIM specifies the time limit per model (and
        # not for the whole run). So, each model stops with a time limit
        # hit, but the whole tuning run just completes without hitting a
        # limit.
        self.assertEqual(status, cpx.parameters.tuning_status.completed)

    def testWithZeroDetTimeLimit(self):
        cpx = self._newCplex()
        cpx.parameters.dettimelimit.set(0)
        status = self.tune(cpx)
        self.assertEqual(status, cpx.parameters.tuning_status.dettime_limit)

    def testWithZeroTuneDetTimeLimit(self):
        cpx = self._newCplex()
        cpx.parameters.tune.dettimelimit.set(0)
        status = self.tune(cpx)
        # See comment in testWithZeroTuneTimeLimit. Same applies here.
        self.assertEqual(status, cpx.parameters.tuning_status.completed)

    def testAbort(self):
        with Aborter() as aborter:
            with self._newCplex() as cpx:
                aborter.abort()
                cpx.use_aborter(aborter)
                status = self.tune(cpx)
                self.assertEqual(status, cpx.parameters.tuning_status.abort)


class TuningTests(CplexTestCase, SharedTuningTests):

    def tune(self, cpx, paramset=None):
        return cpx.parameters.tune_problem(paramset)

    def testWithQCP(self):
        """See SharedTuningTests.testWithQCP"""
        cpx = self._newCplex()
        cpx.variables.add(lb=[0.0]*2)
        cpx.quadratic_constraints.add(
            lin_expr=SparsePair(ind=[0], val=[1.0]),
            quad_expr=SparseTriple(ind1=[0], ind2=[1], val=[1.0]))
        self.assertEqual(cpx.get_problem_type(), cpx.problem_type.QCP)
        status = cpx.parameters.tune_problem()
        self.assertEqual(status, cpx.parameters.tuning_status.completed)

    def testRepeat(self):
        repeat = 2
        rp = RepeatProcessor()
        self.assertEqual(rp.count, 0)
        with self._newCplex() as cpx:
            cpx.parameters.tune.repeat.set(repeat)
            self._setAllStreams(cpx, rp)
            self.tune(cpx)
        self.assertEqual(rp.count, repeat)


class TuneSetTests(CplexTestCase, SharedTuningTests):

    VERYSMALL = "../../data/verysmall.lp"

    def tune(self, cpx, paramset=None):
        return cpx.parameters.tune_problem_set(
            filenames=[self.VERYSMALL], filetypes=None,
            fixed_parameters_and_values=paramset)

    def testWithQCP(self):
        """See SharedTuningTests.testWithQCP"""
        cpx = self._newCplex()
        status = cpx.parameters.tune_problem_set(
            filenames=["../../data/qcp.lp"],
            filetypes=None,
            fixed_parameters_and_values=None)
        self.assertEqual(status, cpx.parameters.tuning_status.completed)

    def testNone(self):
        cpx = self._newCplex()
        try:
            cpx.parameters.tune_problem_set([None])
        except CplexSolverError as err:
            self.assertEqual(err.args[2], error_codes.CPXERR_NULL_POINTER)

    def testBadFileType(self):
        cpx = self._newCplex()
        try:
            cpx.parameters.tune_problem_set([self.VERYSMALL], [".bogus"])
        except CplexSolverError as err:
            self.assertEqual(err.args[2], error_codes.CPXERR_BAD_FILETYPE)

    def testIncompatibleFileType(self):
        cpx = self._newCplex()
        filetypes = [".lp", ".sav", ".mps"]
        compressiontypes = [".gz", ".bz2", ""]
        for filetype in filetypes:
            for comptype in compressiontypes:
                if self.VERYSMALL.endswith(filetype):
                    continue
                try:
                    cpx.parameters.tune_problem_set([self.VERYSMALL],
                                                    [filetype + comptype])
                except CplexSolverError as err:
                    self.assertEqual(err.args[2],
                                     error_codes.CPXERR_BAD_FILETYPE)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
