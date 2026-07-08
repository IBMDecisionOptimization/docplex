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
Tests Cplex.end().

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes

ILLEGAL_METHOD_INVOCATION = 'illegal method invocation after Cplex.end()'

class EndTest(CplexTestCase):

    def testExpectedException(self):
        cpx = self._newCplex()
        cpx.end()
        try:
            cpx.solve()
            self.fail()
        except ValueError as ve:
            self.assertTrue(str(ve), ILLEGAL_METHOD_INVOCATION)

    def testEndIdempotent(self):
        cpx = self._newCplex()
        cpx.end()
        try:
            cpx.end()
        except Error as err:
            self.fail(str(err))

    def testWithClosedFileStream(self):
        with self._getTempFileName(delete=True) as tmp:
            cpx = self._newCplex()
            with open(tmp, 'w') as ftmp:
                self._setAllStreams(cpx, ftmp)
                ftmp.write('Hello, World\n')
                cpx.solve()
            # We should not attempt to do anything with the stream if it's
            # been closed.
            cpx.end()
            # If no exception is thrown we're happy :-)

    def testContextManagerSupport(self):
        with self._newCplex() as cpx:
            cpx.solve()
            self.assertEqual(cpx.solution.status.optimal,
                             cpx.solution.get_status())
        try:
            cpx.solve()
            self.fail()
        except ValueError as ve:
            self.assertTrue(str(ve), ILLEGAL_METHOD_INVOCATION)

    def testSubInterface(self):
        with self._newCplex() as cpx:
            try:
                cpx.solution.get_objective_value()
                self.fail()
            except CplexSolverError as cse:
                self.assertEqual(cse.args[2], error_codes.CPXERR_NO_SOLN)
        try:
            cpx.solution.get_objective_value()
            self.fail()
        except ValueError as ve:
            self.assertTrue(str(ve), ILLEGAL_METHOD_INVOCATION)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
