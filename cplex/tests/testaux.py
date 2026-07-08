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
Tests _aux_functions.py.

No command line arguments are required.
"""
import unittest
import cplex._internal._aux_functions as _aux
from cplex.exceptions import CplexError


class AuxTests(unittest.TestCase):

    def testMaxArgLen(self):
        foo = [0, 1, 2]
        bar = [3, 4, 5]
        num = _aux.max_arg_length([foo, bar])
        self.assertEqual(num, 3)

    def testMaxArgLenWithEmpty(self):
        foo = []
        bar = [0, 1, 2]
        num = _aux.max_arg_length([foo, bar])
        self.assertEqual(num, 3)

    def testValArgLenWithEmpty(self):
        foo = []
        bar = [2, 3, 4]
        _aux.validate_arg_lengths([foo, bar], allow_empty=True)

    def testValArgLenBad(self):
        foo = [0, 1]
        bar = [2, 3, 4]
        try:
            _aux.validate_arg_lengths([foo, bar], allow_empty=True)
            self.assertFalse(__debug__)
        except CplexError as err:
            self.assertIn("inconsistent argument lengths", str(err))

    def testValArgLenDontAllowEmpty(self):
        foo = []
        bar = [0, 1, 2]
        try:
            _aux.validate_arg_lengths([foo, bar], allow_empty=False)
            self.assertFalse(__debug__)
        except CplexError as err:
            self.assertIn("inconsistent argument lengths", str(err))

    def testValArgLenDontAllowEmptyBad(self):
        foo = [0, 1]
        bar = [0, 1, 2]
        try:
            _aux.validate_arg_lengths([foo, bar], allow_empty=False)
            self.assertFalse(__debug__)
        except CplexError as err:
            self.assertIn("inconsistent argument lengths", str(err))

    def testValArgLenDontAllowEmptyGood(self):
        foo = [0, 1, 2]
        bar = [0, 1, 2]
        _aux.validate_arg_lengths([foo, bar], allow_empty=False)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
