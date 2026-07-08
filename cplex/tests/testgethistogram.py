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
Tests get_histogram() and the Histogram class.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase

CASO8_EXAMPLE_FILE = "../../../examples/data/caso8.mps"
LP_EXAMPLE_FILE = "../../../examples/data/lpprog.lp"


class GetHistogramTestCase(CplexTestCase):

    def testEmptyRowCount(self):
        cpx = self._newCplex()
        histo = cpx.linear_constraints.get_histogram()
        histo_str = str(histo)
        expected_str = ''
        self.assertEqual(histo_str, expected_str)

    def testEmptyColumnCount(self):
        cpx = self._newCplex()
        histo = cpx.variables.get_histogram()
        histo_str = str(histo)
        expected_str = ''
        self.assertEqual(histo_str, expected_str)

    def testRowCount(self):
        cpx = self._newCplex()
        cpx.read(CASO8_EXAMPLE_FILE)
        histo = cpx.linear_constraints.get_histogram()
        histo_str = str(histo)
        expected_str = """\
Row counts (excluding fixed variables):

 Nonzero Count:     6     7     8     9    10    11    12    13    14    15
Number of Rows:     4     8     8     4    20    12    36    16     8    20

 Nonzero Count:   393
Number of Rows:   100

"""
        self.assertEqual(histo_str, expected_str)

    def testColumnCount(self):
        cpx = self._newCplex()
        cpx.read(CASO8_EXAMPLE_FILE)
        histo = cpx.variables.get_histogram()
        histo_str = str(histo)
        expected_str = """\
Column counts (excluding fixed variables):

    Nonzero Count:   21   22   24   25   26   27   29   30   52  100  104
Number of Columns:    1  299    1  217    1  217    1  299  224    1   21

"""
        self.assertEqual(histo_str, expected_str)

    def testRowCountByIndex(self):
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        histo = cpx.linear_constraints.get_histogram()
        expected_lst = [0, 0, 0, 3, 0, 0]
        expected_cnt = cpx.variables.get_num() + 1
        self.assertEqual(len(expected_lst), expected_cnt)
        for i in range(expected_cnt):
            self.assertEqual(histo[i], expected_lst[i])
        self.assertEqual(histo[:], expected_lst[:])
        self.assertEqual(histo[1:2], expected_lst[1:2])
        self.assertEqual(histo[::2], expected_lst[::2])

    def testColumnCountByIndex(self):
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        histo = cpx.variables.get_histogram()
        expected_lst = [2, 0, 0, 3]
        expected_cnt = cpx.linear_constraints.get_num() + 1
        self.assertEqual(len(expected_lst), expected_cnt)
        for i in range(expected_cnt):
            self.assertEqual(histo[i], expected_lst[i])
        self.assertEqual(histo[:], expected_lst[:])
        self.assertEqual(histo[1:2], expected_lst[1:2])
        self.assertEqual(histo[::2], expected_lst[::2])

    def testNegativeIndex(self):
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        histo = cpx.linear_constraints.get_histogram()
        try:
            histo[-1]
            self.fail()
        except IndexError as ce:
            self.assertEqual(str(ce), "histogram keys must be non-negative")

    def testNegativeStartIndex(self):
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        histo = cpx.variables.get_histogram()
        try:
            histo[-1:]
            self.fail()
        except IndexError as ce:
            self.assertEqual(str(ce), "histogram keys must be non-negative")

    def testNegativeStopIndex(self):
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        histo = cpx.variables.get_histogram()
        try:
            histo[:-1]
            self.fail()
        except IndexError as ce:
            self.assertEqual(str(ce), "histogram keys must be non-negative")

    def testBadIndexType(self):
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        histo = cpx.variables.get_histogram()
        try:
            histo['foo']
            self.fail()
        except TypeError as te:
            self.assertEqual(str(te), "key must be an integer or a slice")


def main():
    """The main function."""
    unittest.main()

if __name__ == "__main__":
    main()
