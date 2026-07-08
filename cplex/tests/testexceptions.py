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
Tests the cplex.exceptions.CplexError and CplexSolverError classes.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase
from cplex.exceptions import CplexError
from cplex.exceptions import CplexSolverError

EXCEPTION_MSG = 'foo bar bang bat'


class ExceptionTests(CplexTestCase):

    def testCplexError(self):
        try:
            raise CplexError(EXCEPTION_MSG)
            self.fail()
        except CplexError as ce:
            self.assertEqual(str(ce), EXCEPTION_MSG)

    def testCplexSolverError(self):
        env_dummy = 1
        status_dummy = 2
        try:
            raise CplexSolverError(EXCEPTION_MSG, env_dummy, status_dummy)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(str(cse), EXCEPTION_MSG)
            self.assertEqual(cse.args[0], EXCEPTION_MSG)
            self.assertEqual(cse.args[1], env_dummy)
            self.assertEqual(cse.args[2], status_dummy)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
