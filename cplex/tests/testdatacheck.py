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
Tests that datachecks work as expected.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes

NAN_MODEL = "../../data/SPARK-mip.sav.gz"
BAD_SENSE_MODEL = "../../data/vernon.sav.gz"

class DataCheckTests(CplexTestCase):

    def testOnByDefault(self):
        cpx = self._newCplex()
        # The datacheck parameter should be on by default in the
        # Python API
        self.assertEqual(cpx.parameters.read.datacheck.get(),
                         cpx.parameters.read.datacheck.values.warn)

    def testNanModelWithout(self):
        cpx = self._newCplex()
        cpx.parameters.read.datacheck.set(
            cpx.parameters.read.datacheck.values.off)
        cpx.read(NAN_MODEL)
        # NB: We seem to be able to solve this "bad" model fine on
        #     x86-64_linux/static_pic_gcc, but it's probably better not
        #     to invite harm.
        #cpx.solve()
        # Expecting no error

    def testNanModel(self):
        cpx = self._newCplex()
        try:
            cpx.read(NAN_MODEL)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NAN)

    def testBadSenseModelWithout(self):
        cpx = self._newCplex()
        cpx.parameters.read.datacheck.set(
            cpx.parameters.read.datacheck.values.off)
        cpx.read(BAD_SENSE_MODEL)
        # NB: The following triggers an assert in the callable library.
        #     As I understand it, this is probably fine, as we are free
        #     to crash in ugly ways if datacheck is not "on", in order to
        #     maintain performance.
        #cpx.solve()
        # Expecting no error

    def testBadSenseModel(self):
        cpx = self._newCplex()
        try:
            cpx.read(BAD_SENSE_MODEL)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_SENSE)

    def testColIndexRangeWithout(self):
        cpx = self._newCplex()
        cpx.parameters.read.datacheck.set(
            cpx.parameters.read.datacheck.values.off)
        cpx.linear_constraints.add(lin_expr=[[[0], [0]]])
        # Expecting no error

    def testColIndexRange(self):
        cpx = self._newCplex()
        try:
            cpx.linear_constraints.add(lin_expr=[[[0], [0]]])
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_COL_INDEX_RANGE)

    # TODO: Many more tests could be done here.

def main():
    unittest.main()

if __name__ == '__main__':
    main()
