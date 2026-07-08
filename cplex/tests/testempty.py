# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Tests solve() on empty models and checks exported LP file.

No command line arguments are required.
"""
import unittest
import os
import cplex
from cplextestcase import CplexTestCase

class EmptyTests(CplexTestCase):

    def testEmpty(self):
        cpx = self._newCplex()
        expected_lp = r"""\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1:
End
"""
        self._solveAndExport(cpx, expected_lp)

    def testEmptyWithObjName(self):
        cpx = self._newCplex()
        cpx.objective.set_name("empty")
        expected_lp = r"""\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 _empty#0:
End
"""
        self._solveAndExport(cpx, expected_lp)

    def testEmptyWithRowName(self):
        cpx = self._newCplex()
        cpx.objective.set_name("empty_obj")
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.linear_constraints.add(names = ["empty_row"])
        expected_lp = r"""\ENCODING=ISO-8859-1
\Problem name: 

Maximize
 _empty_obj#0:
Subject To
 _empty_row#0:  = 0
End
"""
        self._solveAndExport(cpx, expected_lp)

    def testEmptyWithColName(self):
        cpx = self._newCplex()
        cpx.variables.add(names = ["empty_col"], obj = [1], types = "I")
        expected_lp = r"""\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: _empty_col#0
Bounds
      _empty_col#0 >= 0
Generals
 _empty_col#0 
End
"""
        self._solveAndExport(cpx, expected_lp)

    def testEmptyNoObjWithColName(self):
        cpx = self._newCplex()
        cpx.variables.add(names = ["empty_col"], types = "I")
        expected_lp = r"""\ENCODING=ISO-8859-1
\Problem name: 

Minimize
 obj1: 0 _empty_col#0
Bounds
      _empty_col#0 >= 0
Generals
 _empty_col#0 
End
"""
        self._solveAndExport(cpx, expected_lp)

    def testEmptyWithProblemName(self):
        cpx = self._newCplex()
        cpx.set_problem_name("empty")
        expected_lp = r"""\ENCODING=ISO-8859-1
\Problem name: empty

Minimize
 obj1:
End
"""
        self._solveAndExport(cpx, expected_lp)


    def _solveAndExport(self, cpx, expected_lp):
        lp_file = "empty.lp"
        self._failSafeDelete(lp_file)
        # Solve empty model.
        cpx.solve()
        # Export LP file and compare with expected contents.
        cpx.write(lp_file)
        self.assertTrue(os.path.isfile(lp_file))
        actual_lp = self._readFile(lp_file)
        self.assertEqual(actual_lp, expected_lp)
        self._failSafeDelete(lp_file)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
