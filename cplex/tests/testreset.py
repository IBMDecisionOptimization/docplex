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
Tests parameters.reset.

No command line arguments are required.
"""
import unittest
import cplex
from cplex.exceptions import CplexError, CplexSolverError, error_codes
from cplex._internal import _constants
from cplextestcase import CplexTestCase

EXAMPLE_FILE = "../../../examples/data/afiro.mps"
EXAMPLE_FILE_TYPE = "mps"

class ResetTests(CplexTestCase):

    def testOverriddenDefaults(self):
        cpx = self._newCplex()
        self.checkDefaults(cpx)

    def testFixedParams(self):
        cpx = self._newCplex()
        for paramid, paramtype in (
                (_constants.CPX_PARAM_APIENCODING,
                 _constants.CPX_PARAMTYPE_STRING),
                (_constants.CPX_PARAM_MIPCBREDLP,
                 _constants.CPX_PARAMTYPE_INT)):
            try:
                cpx.parameters._get(paramid, paramtype)
                self.fail()
            except CplexSolverError as cse:
                self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_PARAM_NUM)

    def testResetFromDefaults(self):
        cpx = self._newCplex()
        cpx.parameters.reset()
        self.checkDefaults(cpx)

    def testResetFromChange(self):
        cpx = self._newCplex()
        self.assertEqual(cpx.parameters.advance.get(),
                         cpx.parameters.advance.default())
        cpx.parameters.advance.set(0)
        self.assertNotEqual(cpx.parameters.advance.get(),
                            cpx.parameters.advance.default())
        cpx.parameters.reset()
        self.assertEqual(cpx.parameters.advance.get(),
                         cpx.parameters.advance.default())

    @staticmethod
    def getTuningParams(cpx):
        return [(cpx.parameters.lpmethod,
                 cpx.parameters.lpmethod.values.barrier),
                (cpx.parameters.read.datacheck,
                 cpx.parameters.read.datacheck.values.off)]

    def checkTuningParams(self, cpx):
        self.assertEqual(cpx.parameters.lpmethod.get(),
                         cpx.parameters.lpmethod.values.barrier)
        self.assertNotEqual(cpx.parameters.lpmethod.get(),
                            cpx.parameters.lpmethod.default())
        self.assertEqual(cpx.parameters.read.datacheck.get(),
                         cpx.parameters.read.datacheck.values.off)
        self.assertNotEqual(cpx.parameters.read.datacheck.get(),
                            cpx.parameters.read.datacheck.default())

    def checkResetAfterTuning(self, cpx):
        cpx.parameters.reset()
        self.assertEqual(cpx.parameters.lpmethod.get(),
                         cpx.parameters.lpmethod.default())
        self.checkDefaults(cpx)

    def testResetFromTune(self):
        cpx = self._newCplex()
        cpx.parameters.tune_problem(self.getTuningParams(cpx))
        self.checkTuningParams(cpx)
        self.checkResetAfterTuning(cpx)

    def testResetFromTuneSet(self):
        cpx = self._newCplex()
        cpx.parameters.tune_problem_set(
            [EXAMPLE_FILE],
            [EXAMPLE_FILE_TYPE],
            self.getTuningParams(cpx))
        self.checkTuningParams(cpx)
        self.checkResetAfterTuning(cpx)

    def checkDefaults(self, cpx):
        # After calling reset, check that parameters with special
        # treatment in the Python API are set properly.
        self.assertEqual(cpx.parameters.read.datacheck.get(),
                         cpx.parameters.read.datacheck.values.warn)
        self.assertEqual(cpx.parameters.output.clonelog.min(), 0)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
