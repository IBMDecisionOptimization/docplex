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
Tests get_version and get_versionnumber.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase


class VersionTests(CplexTestCase):

    def testVersion(self):
        expectedVersion = self.getVersionString()
        cpx = self._newCplex()
        actualVersion = cpx.get_version()
        # We check startswith here, rather than equals, because actual may
        # contain information extra information, such as the output of
        # util/gitid.
        self.assertTrue(actualVersion.startswith(expectedVersion))

    def testVersionNumber(self):
        (v, r, m, f) = self.getVersionInfo()
        cpx = self._newCplex()
        self.assertEqual(f + 100 * m + 10000 * r + 1000000 * v,
                         cpx.get_versionnumber())

    def testVersionAttribute(self):
        self.assertEqual(cplex.__version__,
                         self.getVersionString())

    def getVersionString(self):
        (v, r, m, f) = self.getVersionInfo()
        return "%s.%s.%s.%s" % (v, r, m, f)

    def getVersionInfo(self):
        v = cplex._internal._pycplex.CPX_VERSION_VERSION
        r = cplex._internal._pycplex.CPX_VERSION_RELEASE
        m = cplex._internal._pycplex.CPX_VERSION_MODIFICATION
        f = cplex._internal._pycplex.CPX_VERSION_FIX
        return (v, r, m, f)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
