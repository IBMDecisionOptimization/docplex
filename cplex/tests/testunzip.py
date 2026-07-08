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
Tests unzip function.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex._internal._aux_functions import unzip

# Shared test data
L1 = [1, 2, 3]
L2 = [4, 5, 6]
L3 = [7, 8, 9]
S1 = 'abc'
S2 = 'def'


class UnzipTests(CplexTestCase):

    def testEmptyList(self):
        self.assertEqual([], unzip([]))

    def testNoArg(self):
        self.assertEqual([], unzip())

    def testList(self):
        z = list(zip(L1, L2))
        self.assertEqual([tuple(L1), tuple(L2)], unzip(z))

    def testTuple(self):
        z = tuple(zip(L1, L2))
        self.assertEqual([tuple(L1), tuple(L2)], unzip(z))

    def testIterator(self):
        z = zip(L1, L2)
        self.assertEqual([tuple(L1), tuple(L2)], unzip(z))

    def testGenerator(self):
        z = ((i, j) for i, j in zip(L1, L2))
        self.assertEqual([tuple(L1), tuple(L2)], unzip(z))

    def testTriple(self):
        z = zip(L1, L2, L3)
        self.assertEqual([tuple(L1), tuple(L2), tuple(L3)], unzip(z))

    def testStrings(self):
        z = zip(S1, S2)
        self.assertEqual([tuple(S1), tuple(S2)], unzip(z))

    def testMixed(self):
        z = zip(L1, S1)
        self.assertEqual([tuple(L1), tuple(S1)], unzip(z))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
