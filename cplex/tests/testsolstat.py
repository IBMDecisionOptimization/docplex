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
Tests the SolutionStatus class.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase


class SolutionStatusTests(CplexTestCase):

    def testCount(self):
        solstatdict = self._get_solstat_dict()
        solstatattrs = self._get_solstat_attrs(solstatdict)
        solstatcnt = len(solstatattrs)
        constdict = self._get_const_dict()
        constnames = self._get_solstat_const_names(constdict)
        constcnt = len(constnames)
        self.assertEqual(constcnt, solstatcnt)

    def testGetItem(self):
        solstat = self._get_solstat()
        constdict = self._get_const_dict()
        constnames = self._get_solstat_const_names(constdict)
        for key in constnames:
            constval = constdict[key]
            constrepr = solstat[constval]
            # SolutionStatus.__getitem__ should return string representations
            # of the Python "constants".
            self.assertFalse(constrepr is None,
                        "{0}={1} not found in SolutionStatus"
                        .format(key, constval))

    def testStatusInConst(self):
        solstat = self._get_solstat()
        solstatdict = self._get_solstat_dict()
        solstatattrs = self._get_solstat_attrs(solstatdict)
        constdict = self._get_const_dict()
        constnames = self._get_solstat_const_names(constdict)
        for skey in solstatattrs:
            sval = solstatdict[skey]
            if sval == 0:
                # There is no constant for "Unknown status value".
                continue
            for ckey in constnames:
                cval = constdict[ckey]
                if sval == cval:
                    break
            else:
                self.fail("Failed to find {0}!".format(skey))

    def testConstInStatus(self):
        solstat = self._get_solstat()
        solstatdict = self._get_solstat_dict()
        solstatattrs = self._get_solstat_attrs(solstatdict)
        constdict = self._get_const_dict()
        constnames = self._get_solstat_const_names(constdict)
        for ckey in constnames:
            cval = constdict[ckey]
            for skey in solstatattrs:
                sval = solstatdict[skey]
                if cval == sval:
                    break
            else:
                self.fail("Failed to find {0}!".format(ckey))

    @staticmethod
    def _get_solstat():
        return cplex._internal._subinterfaces.SolutionStatus()

    @staticmethod
    def _get_solstat_dict():
        return cplex._internal._subinterfaces.SolutionStatus.__dict__

    @staticmethod
    def _get_solstat_attrs(solstatdict):
        # We ignore anything that is private (i.e., starts with "_") or is a
        # magic method (i.e., starts with "__").
        return [attr
                for attr
                in solstatdict.keys()
                if not attr.startswith('_')]

    @staticmethod
    def _get_const_dict():
        return cplex._internal._constants.__dict__

    @staticmethod
    def _get_solstat_const_names(constdict):
        # This is a little bit of a hack.  We depend on the consistent naming
        # of the solution status codes.  For now, it should suffice, though.
        return [const
                for const
                in constdict.keys()
                if const.startswith('CPX_STAT')
                or const.startswith('CPXMIP')]


def main():
    unittest.main()

if __name__ == '__main__':
    main()
