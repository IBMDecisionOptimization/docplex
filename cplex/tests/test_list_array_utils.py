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
Tests the stuff in cplex._internal._list_array_utils

No command line arguments are required.
"""
import unittest
import sys
import cplex
from cplextestcase import CplexTestCase
from cplex._internal._list_array_utils import (
    int_list_to_array,
    int_list_to_array_trunc_int32,
    long_list_to_array,
    double_list_to_array,
    array_to_list,
    int_c_array,
    long_c_array,
    double_c_array)
from cplex._internal import _pycplex as _CPX
CPX_NULL = _CPX.cvar.CPX_NULL


class ListArrayUtilTests(CplexTestCase):

    def testListToArrayTruncInt32(self):
        int32_min = -2147483648
        int32_max = 2147483647
        tpl = (0, 1, int32_min - 1, int32_max + 1)
        result = int_list_to_array_trunc_int32(tpl)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)
        self.assertEqual(result[2], int32_min)
        self.assertEqual(result[3], int32_max)

    def testListToArrayTruncInt32Empty(self):
        result = int_list_to_array_trunc_int32(tuple())
        self.assertEqual(result, CPX_NULL)


    def testIntListToArray(self):
        lst_in = [0, 1, 2]
        ary = int_list_to_array(lst_in)
        lst_out = array_to_list(ary, len(lst_in))
        self.assertEqual(lst_in, lst_out)

    def testLongListToArray(self):
        lst_in = [0, 1, 2]
        ary = long_list_to_array(lst_in)
        lst_out = array_to_list(ary, len(lst_in))
        self.assertEqual(lst_in, lst_out)

    def testDoubleListToArray(self):
        lst_in = [0.0, 1.0, 2.0]
        ary = double_list_to_array(lst_in)
        lst_out = array_to_list(ary, len(lst_in))
        self.assertEqual(lst_in, lst_out)


class IntCArrayTests(CplexTestCase):

    def testEmpty(self):
        with int_c_array([]) as ary:
            self.assertIsInstance(ary, int)

    def testSimple(self):
        with int_c_array([0, 1, 2]) as ary:
            self.assertIsInstance(ary, int)

    def testNonIntegral(self):
        with self.assertRaises(TypeError) as cm:
            with int_c_array('blah'):
                pass
        self.assertIn('non-integral value in input sequence',
                      str(cm.exception))

    def testNonSequence(self):
        with self.assertRaises(TypeError) as cm:
            with int_c_array(0):
                pass
        self.assertIn('argument must be a sequence',
                      str(cm.exception))

    def testLongItem(self):
        if CplexTestCase.iswindows():
            # On 64-bit Windows, we will trigger an OverflowError
            # because a 'long' C type is always a 32-bit signed integer.
            with self.assertRaises(OverflowError):
                with int_c_array([sys.maxsize]):
                    pass
        else:
            with self.assertRaises(ValueError) as cm:
                with int_c_array([sys.maxsize]):
                    pass
            self.assertIn('long value in input sequence', str(cm.exception))

    def testOverflow(self):
        with self.assertRaises(OverflowError):
            with int_c_array([sys.maxsize + 1]):
                pass


class LongCArrayTests(CplexTestCase):

    def testEmpty(self):
        with long_c_array([]) as ary:
            self.assertIsInstance(ary, int)

    def testSimple(self):
        with long_c_array([0, 1, 2]) as ary:
            self.assertIsInstance(ary, int)

    def testNonIntegral(self):
        with self.assertRaises(TypeError) as cm:
            with long_c_array('blah'):
                pass
        self.assertIn('non-integral value in input sequence',
                      str(cm.exception))

    def testNonSequence(self):
        with self.assertRaises(TypeError) as cm:
            with long_c_array(0):
                pass
        self.assertIn('argument must be a sequence', str(cm.exception))

    def testLongItem(self):
        with long_c_array([sys.maxsize]):
            pass

    def testOverflow(self):
        with self.assertRaises(OverflowError):
            with long_c_array([sys.maxsize + 1]):
                pass


class DoubleCArrayTests(CplexTestCase):

    def testEmpty(self):
        with double_c_array([]) as ary:
            self.assertIsInstance(ary, int)

    def testSimple(self):
        with double_c_array([0.0, 1.0, 2.0]) as ary:
            self.assertIsInstance(ary, int)

    def testIntegral(self):
        with double_c_array([0, 1, 2]) as ary:
            self.assertIsInstance(ary, int)

    def testNonFloat(self):
        with self.assertRaises(TypeError) as cm:
            with double_c_array('blah'):
                pass
        self.assertIn('non-float value in input sequence',
                      str(cm.exception))

    def testNonSequence(self):
        with self.assertRaises(TypeError) as cm:
            with double_c_array(0):
                pass
        self.assertIn('argument must be a sequence',
                      str(cm.exception))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
