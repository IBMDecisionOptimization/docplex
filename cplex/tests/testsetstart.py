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
Tests the Cplex.start.set_start method.

No command line arguments are required.
"""
import unittest
from cplextestcase import CplexTestCase

AFIRO = '../../data/afiro.mps'

class SetStartTests(CplexTestCase):

    @classmethod
    def setUpClass(cls):
        cls.cpx1 = cls._newCplex()
        cls.cpx1.read(AFIRO)
        detstart = cls.cpx1.get_dettime()
        cls.cpx1.solve()
        cls.dettime1 = cls.cpx1.get_dettime() - detstart
        cls.numiter1 = cls.cpx1.solution.progress.get_num_iterations()
        cls.obj1 = cls.cpx1.solution.get_objective_value()
        with cls._getTempFileName(ext='.bas', delete=False) as tmp:
            cls.basisfile = tmp

    @classmethod
    def tearDownClass(cls):
        cls._failSafeDelete(cls.basisfile)

    def setUp(self):
        self.assertEqual(self.cpx1.get_problem_type(),
                         self.cpx1.problem_type.LP)
        self.assertEqual(self.cpx1.solution.status.optimal,
                         self.cpx1.solution.get_status())

    def check(self, cpx2):
        """Solves the second model (with start info) and checks it."""
        detstart = cpx2.get_dettime()
        cpx2.solve()

        dettime2 = cpx2.get_dettime() - detstart
        self.assertGreater(self.dettime1, dettime2)

        self.assertGreater(self.numiter1,
                           cpx2.solution.progress.get_num_iterations())

        self.assertAlmostEqual(self.obj1,
                               cpx2.solution.get_objective_value())

    def testBasis(self):
        col_basis, row_basis = self.cpx1.solution.basis.get_basis()
        cpx2 = self._newCplex()
        cpx2.read(AFIRO)
        cpx2.start.set_start(col_status=col_basis,
                             row_status=row_basis,
                             col_primal=[], row_primal=[],
                             col_dual=[], row_dual=[])
        self.check(cpx2)

    def testPrimalAndDuals(self):
        cpx2 = self._newCplex()
        cpx2.read(AFIRO)
        cpx2.start.set_start(
            col_status=[], row_status=[],
            col_primal=self.cpx1.solution.get_values(),
            row_primal=self.cpx1.solution.get_linear_slacks(),
            col_dual=self.cpx1.solution.get_reduced_costs(),
            row_dual=self.cpx1.solution.get_dual_values())
        self.check(cpx2)

    def testWriteReadBasis(self):
        self.cpx1.solution.basis.write(self.basisfile)

        cpx2 = self._newCplex()
        cpx2.read(AFIRO)
        cpx2.start.read_basis(self.basisfile)
        self.check(cpx2)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
