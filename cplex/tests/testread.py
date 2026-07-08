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
Tests the Cplex.read() method.

No command line arguments are required.
"""
import unittest
import os.path
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase

BAD_EXAMPLE_FILE = '../../../examples/foo/bar.lp'
LP_EXAMPLE_FILE = '../../../examples/data/case1.lp'
LPGZ_EXAMPLE_FILE = '../../../examples/data/flugpl.lp.gz'
MPS_EXAMPLE_FILE = '../../../examples/data/caso8.mps'
MPSGZ_EXAMPLE_FILE = '../../../examples/data/case2.mps.gz'
SAV_EXAMPLE_FILE = '../../../examples/data/flow2ez.sav'
SAVGZ_EXAMPLE_FILE = '../../../examples/data/aflo.sav.gz'
BAD_FILE_TYPE = 'bogus'
# TODO: It would be nice if these file types were available in the Cplex object
#       so that it would be self-documented what file types are available.
LP_FILE_TYPE = 'LP'
MPS_FILE_TYPE = 'MPS'
SAV_FILE_TYPE = 'SAV'

class ReadTests(CplexTestCase):

    def testEmptyString(self):
        cpx = self._newCplex()
        try:
            cpx.read('')
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_ARGUMENT)

    def testNone(self):
        cpx = self._newCplex()
        try:
            cpx.read(None)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NULL_POINTER)

    def testBadPathWithImplicitFileType(self):
        cpx = self._newCplex()
        self._testBadPath(lambda s: cpx.read(s))

    def testBadPathWithExplicitFileType(self):
        cpx = self._newCplex()
        self._testBadPath(lambda s: cpx.read(s, BAD_FILE_TYPE))

    def _testBadPath(self, func):
        try:
            self.assertFalse(os.path.isfile(BAD_EXAMPLE_FILE))
            func(BAD_EXAMPLE_FILE)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_FAIL_OPEN_READ)
            self.assertTrue("'{0}'".format(BAD_EXAMPLE_FILE) in str(cse))

    def testInvalidFileType(self):
        cpx = self._newCplex()
        try:
            cpx.read(LP_EXAMPLE_FILE, BAD_FILE_TYPE)
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_FILETYPE)

    def testSavAsLP(self):
        """This test attempts to read in a SAV file as an LP file.

        We expect it to fail to find an objective sense.
        """
        cpx = self._newCplex()
        try:
            cpx.read(SAV_EXAMPLE_FILE, LP_FILE_TYPE)
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_OBJ_SENSE)

    def testNoNameSection(self):
        cpx = self._newCplex()
        try:
            cpx.read(LP_EXAMPLE_FILE, MPS_FILE_TYPE)
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_NAME_SECTION)

    def testNotSavFile(self):
        cpx = self._newCplex()
        try:
            cpx.read(LP_EXAMPLE_FILE, SAV_FILE_TYPE)
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_SAV_FILE)

    def testLp(self):
        self._testImpliedFileType(LP_EXAMPLE_FILE)
        self._testImpliedFileType(LPGZ_EXAMPLE_FILE)
        self._testExplicitFileType(LP_EXAMPLE_FILE, LP_FILE_TYPE)
        self._testExplicitFileType(LPGZ_EXAMPLE_FILE, LP_FILE_TYPE)

    def testMps(self):
        self._testImpliedFileType(MPS_EXAMPLE_FILE)
        self._testImpliedFileType(MPSGZ_EXAMPLE_FILE)
        self._testExplicitFileType(MPS_EXAMPLE_FILE, MPS_FILE_TYPE)
        self._testExplicitFileType(MPSGZ_EXAMPLE_FILE, MPS_FILE_TYPE)

    def testSav(self):
        self._testImpliedFileType(SAV_EXAMPLE_FILE)
        self._testImpliedFileType(SAVGZ_EXAMPLE_FILE)
        self._testExplicitFileType(SAV_EXAMPLE_FILE, SAV_FILE_TYPE)
        self._testExplicitFileType(SAVGZ_EXAMPLE_FILE, SAV_FILE_TYPE)

    def _testImpliedFileType(self, filename):
        self._testExplicitFileType(filename, '')

    def _testExplicitFileType(self, filename, filetype):
        cpx = self._newCplex()
        self.assertEqual(cpx.get_problem_name(), '')
        cpx.read(filename, filetype)
        self.assertEqual(cpx.get_problem_name(), filename)

def main():
    unittest.main()


if __name__ == '__main__':
    main()
