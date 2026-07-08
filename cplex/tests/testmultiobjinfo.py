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
Tests the MultiObj*Info classes.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex._internal._multiobjsoln import (MultiObjIntInfo,
                                           MultiObjLongInfo,
                                           MultiObjFloatInfo)


class MultiObjInfoTests(CplexTestCase):

    def testCount(self):
        """Check that the number of CPX_MULTIOBJ_* macros equals the total
        number of "public" MultiObj*Info attributes.
        """
        int_attrs = MultiObjInfoTests.get_attrs(MultiObjIntInfo.__dict__)
        long_attrs = MultiObjInfoTests.get_attrs(MultiObjLongInfo.__dict__)
        float_attrs = MultiObjInfoTests.get_attrs(MultiObjFloatInfo.__dict__)
        multiobjinfocnt = len(int_attrs) + len(long_attrs) + len(float_attrs)
        constdict = MultiObjInfoTests.get_const_dict()
        constnames = MultiObjInfoTests.get_multiobj_const_names(constdict)
        constcnt = len(constnames)
        self.assertEqual(constcnt, multiobjinfocnt)

    def checkGetItem(self, info_cls):
        """Check that we return strings for each attribute.

        See the __getitem__ method of MultiObj*Info.
        """
        info_obj = info_cls()
        info_attrs = MultiObjInfoTests.get_attrs(info_cls.__dict__)
        for attr in info_attrs:
            try:
                info_obj[getattr(info_obj, attr)]
            except KeyError:
                self.fail("{0}.__getitem__ does not account for '{1}'".format(
                    info_cls, attr))

    def testIntGetItem(self):
        self.checkGetItem(MultiObjIntInfo)

    def testLongGetItem(self):
        self.checkGetItem(MultiObjLongInfo)

    def testFloatGetItem(self):
        self.checkGetItem(MultiObjFloatInfo)

    def testFindMacro(self):
        """Check that for every CPX_MULTIOBJ_* macro, we find it in one
        of the MultiObj*Info classes.
        """
        constdict = MultiObjInfoTests.get_const_dict()
        constnames = MultiObjInfoTests.get_multiobj_const_names(constdict)
        for ckey in constnames:
            cval = constdict[ckey]
            if (cval in MultiObjIntInfo.__dict__.values() or
                cval in MultiObjLongInfo.__dict__.values() or
                cval in MultiObjFloatInfo.__dict__.values()):
                pass
            else:
                self.fail("Failed to find {0}!".format(ckey))

    @staticmethod
    def get_attrs(multiobjinfodict):
        # We ignore anything that is private (i.e., starts with "_") or is a
        # magic method (i.e., starts with "__").
        return [attr
                for attr
                in multiobjinfodict.keys()
                if not attr.startswith('_')]

    @staticmethod
    def get_const_dict():
        return cplex._internal._constants.__dict__

    @staticmethod
    def get_multiobj_const_names(constdict):
        # This is a little bit of a hack.  We depend on the consistent naming
        # of the macros. For now, it should suffice, though.
        return [const
                for const
                in constdict.keys()
                if const.startswith('CPX_MULTIOBJ_')]


def main():
    unittest.main()

if __name__ == '__main__':
    main()
