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
Tests parameter boundaries.

No command line arguments are required.
"""
import unittest
import sys
import cplex
from cplex.exceptions import CplexError, CplexSolverError, error_codes
import cplex._internal._parameter_classes as PC
import cplex._internal._parameters_auto as PA
import cplex._internal._constants as _constants
from cplextestcase import CplexTestCase

MAX_DEPTH = 5


def getparams(cpx):
    for param, _ in cpx.parameters.get_all():
        # Skip the recording parameter as turning it off has no
        # effect and it slows down this test considerably.
        if param._id == _constants.CPX_PARAM_RECORD:
            continue
        yield param


class ParamTests(CplexTestCase):

    numeric_types = [float, int]

    def testIncompatibleTuningTimeLimitParams(self):
        """Ensure that tuning time limit params are not compatible.

        CPX_PARAM_TUNINGDETTILIM is not compatible with
        CPX_PARAM_TUNINGTILIM. Any attempt to set either of these
        parameters to a finite value while the other is already set to a
        finite value should result in the error CPXERR_PARAM_INCOMPATIBLE.
        """
        cpx = self._newCplex()
        cpx.parameters.tune.timelimit.set(1)
        try:
            cpx.parameters.tune.dettimelimit.set(1)
        except CplexSolverError as err:
            self.assertEqual(err.args[2],
                             error_codes.CPXERR_PARAM_INCOMPATIBLE)
        # Now try the other way.
        cpx.parameters.reset()
        cpx.parameters.tune.dettimelimit.set(1)
        try:
            cpx.parameters.tune.timelimit.set(1)
        except CplexSolverError as err:
            self.assertEqual(err.args[2],
                             error_codes.CPXERR_PARAM_INCOMPATIBLE)

    def testChangedParamsOnEmpty(self):
        cpx = self._newCplex()
        changed = cpx.parameters.get_changed()
        for param, value in changed:
            if param is cpx.parameters.parallel:
                # This is not done explicitly here but can be done via
                # the ILOG_CPLEX_PARAMETER_FILE environment variable.
                self.assertEqual(
                    value, cpx.parameters.parallel.values.opportunistic)
            elif param is cpx.parameters.record:
                # This is not done explicitly here but can be done via
                # the ILOG_CPLEX_PARAMETER_FILE environment variable.
                self.assertEqual(value, cpx.parameters.record.values.on)
            else:
                self.fail("Unexpected parameter!")

    def testChangedParams(self):
        cpx = self._newCplex()
        cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.dual)
        changed = cpx.parameters.get_changed()
        found_lpmethod = False
        for param, value in changed:
            if param is cpx.parameters.lpmethod:
                found_lpmethod = True
                self.assertEqual(value, cpx.parameters.lpmethod.values.dual)
            elif param is cpx.parameters.parallel:
                # This is not done explicitly here but can be done via
                # the ILOG_CPLEX_PARAMETER_FILE environment variable.
                self.assertEqual(
                    value, cpx.parameters.parallel.values.opportunistic)
            elif param is cpx.parameters.record:
                # This is not done explicitly here but can be done via
                # the ILOG_CPLEX_PARAMETER_FILE environment variable.
                self.assertEqual(value, cpx.parameters.record.values.on)
            else:
                self.fail("Unexpected parameter!")
        self.assertTrue(found_lpmethod)
        # Now, reset and confirm there are no changed params.
        cpx.parameters.reset()
        changed = cpx.parameters.get_changed()
        self.assertEqual(len(changed), 0)

    def testNumericParams(self):
        cpx = self._newCplex()
        for param in getparams(cpx):
            if param.type() == int:
                self._testIntParamWithDiffTypes(param)
            elif param.type() == float:
                self._testDblParamWithDiffTypes(param)
            else:
                self.assertTrue(isinstance(param.default(), str))

    def _testDblParamWithDiffTypes(self, param):
        self.assertEqual(param.type(), float)
        if (abs(param.max() - param.min()) < 1):
            return
        max_ = param.max()
        default_ = param.default()
        for typ in (int,):
            param.set(typ(max_))
            self.assertEqual(typ(param.get()), typ(max_))
        param.set(float(default_))
        self.assertEqual(param.get(), default_)

    def _testIntParamWithDiffTypes(self, param):
        self.assertEqual(param.type(), int)
        default_ = param.default()
        for typ in self.numeric_types:
            param.set(typ(default_))
            self.assertEqual(param.get(), default_)

    def testInvalidArgument(self):
        cpx = self._newCplex()
        for param in getparams(cpx):
            if param.type() in self.numeric_types:
                self._setNumParamToString(param)
            else:
                self.assertTrue(isinstance(param.default(), str))
                self._setStringParamToNum(param)

    def _setNumParamToString(self, param):
        for x in ("foo", None):
            with self.assertRaises(TypeError):
                param.set(x)

    def _setStringParamToNum(self, param):
        for typ in self.numeric_types:
            with self.assertRaises(TypeError):
                param.set(typ())

    def testBoundaries(self):
        cpx = self._newCplex()
        for param in getparams(cpx):
            if param.type() == int:
                self._testIntParam(param)
            elif param.type() == float:
                self._testDblParam(param)
            else:
                self.assertTrue(isinstance(param.default(), str))
                self._testStrParam(param)

    def _testStrParam(self, param):
        default_ = param.default()
        try:
            param.set(default_)
            self.assertEqual(param.get(), default_)
        except CplexError as ce:
            self.assertEqual(param._id, _constants.CPX_PARAM_CPUMASK)
            self.assertEqual(ce.args[2],
                             error_codes.CPXERR_UNSUPPORTED_OPERATION)
        with self.assertRaises(CplexError) as cm:
            param.set(None)
        self.assertEqual(cm.exception.args[2],
                         error_codes.CPXERR_NULL_POINTER)

    def _testDblParam(self, param):
        min_ = param.min()
        max_ = param.max()
        default_ = param.default()
        param.set(min_)
        self.assertEqual(param.get(), min_)
        param.set(max_)
        self.assertEqual(param.get(), max_)
        # If min == max, then any value must be accepted.
        if min_ == max_:
            param.set(-sys.float_info.max)
            param.set(sys.float_info.max)
        else:
            # Test setting beyond boundaries.
            try:
                param.set(-sys.float_info.max)
                self.fail()
            except CplexError as ce:
                self.assertEqual(ce.args[2],
                                 error_codes.CPXERR_PARAM_TOO_SMALL)
            try:
                param.set(sys.float_info.max)
                self.fail()
            except CplexError as ce:
                self.assertEqual(ce.args[2], error_codes.CPXERR_PARAM_TOO_BIG)
        # Finish, by re-setting to default value.
        param.set(default_)
        self.assertEqual(param.get(), default_)

    def _testIntParam(self, param):
        min_ = param.min()
        max_ = param.max()
        default_ = param.default()
        for value in (min_, max_, default_):
            param.set(value)
            self.assertEqual(param.get(), value)

        try:
            param.set(min_ - 1)
            self.fail()
        except ValueError:
            # CPX_PARAM_CLONELOG has a special min value for the Python API, so
            # we have special handling for it.
            self.assertEqual(param._id, _constants.CPX_PARAM_CLONELOG)
        except CplexError as ce:
            self.assertEqual(ce.args[2],
                             error_codes.CPXERR_PARAM_TOO_SMALL)
        try:
            param.set(max_ + 1)
            self.fail()
        except CplexError as ce:
            self.assertEqual(ce.args[2], error_codes.CPXERR_PARAM_TOO_BIG)

    def test_internal_set(self):
        """Test with and without optional paramtype argument

        See RTC-34595.
        """
        cpx = self._newCplex()
        origval = cpx.parameters.read.datacheck.get()
        newval = _constants.CPX_DATACHECK_OFF
        self.assertNotEqual(origval, newval)
        cpx.parameters._set(_constants.CPX_PARAM_DATACHECK, newval,
                            paramtype=_constants.CPX_PARAMTYPE_INT)
        self.assertEqual(cpx.parameters.read.datacheck.get(), newval)
        cpx.parameters.read.datacheck.set(origval)
        self.assertEqual(cpx.parameters.read.datacheck.get(), origval)
        cpx.parameters._set(_constants.CPX_PARAM_DATACHECK, newval)
        self.assertEqual(cpx.parameters.read.datacheck.get(), newval)

    def test_internal_get(self):
        """Test with and without optional paramtype argument

        See RTC-34595.
        """
        cpx = self._newCplex()
        self.assertEqual(
            cpx.parameters._get(_constants.CPX_PARAM_DATACHECK,
                                paramtype=_constants.CPX_PARAMTYPE_INT),
            cpx.parameters._get(_constants.CPX_PARAM_DATACHECK))

    def test_internal_get_info(self):
        """Test with and without optional paramtype argument

        See RTC-34595.
        """
        cpx = self._newCplex()
        self.assertEqual(cpx.parameters._get_info(
            _constants.CPX_PARAM_DATACHECK,
            paramtype=_constants.CPX_PARAMTYPE_INT),
                         cpx.parameters._get_info(
            _constants.CPX_PARAM_DATACHECK))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
