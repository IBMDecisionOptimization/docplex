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
Tests some methods in the presolve interface.

This was cribbed from the adpreex1.c example and then ported to the
unittest framework.
"""
import unittest
from cplextestcase import CplexTestCase


class AdPreEx1Tests(CplexTestCase):

    def testadpreex1(self):
        # Read the problem file.
        cpx = self._newCplex()
        cpx.read('../../data/prod.lp')

        # Tell presolve to do only primal reductions.
        cpx.parameters.preprocessing.reduce.set(
            cpx.parameters.preprocessing.reduce.values.primal)

        # Turn off simplex logging.
        cpx.parameters.simplex.display.set(
            cpx.parameters.simplex.display.values.none)

        # Optimize the problem and obtain the solution.
        cpx.solve()

        self.assertEqual(cpx.solution.status.optimal,
                         cpx.solution.get_status())

        objval = cpx.solution.get_objective_value()
        self.assertAlmostEqual(532617.0, objval, places=0)

        colnames = cpx.variables.get_names()
        colindices = cpx.variables.get_indices(colnames)
        invindices = [i for i, j in zip(colindices, colnames)
                      if j.startswith('inv')]
        totinv = sum(cpx.solution.get_values(invindices))

        # Check inventory level under profit objective.
        self.assertAlmostEqual(17330.0, totinv, places=0)

        # Get profit objective and add it as a constraint.
        rrhs = objval - abs(objval) * 1e-6
        cpx.presolve.add_rows(lin_expr=[[colindices,
                                         cpx.objective.get_linear()]],
                              senses='G', rhs=[rrhs])

        # Set up objective to maximize negative of sum of inventory.
        values = [-1.0 if idx in invindices else 0.0 for idx in colindices]
        cpx.presolve.set_objective(objective=[colindices, values])

        cpx.solve()

        self.assertEqual(cpx.solution.status.optimal,
                         cpx.solution.get_status())

        objval = cpx.solution.get_objective_value()
        # Check inventory level after optimization.
        self.assertAlmostEqual(-13729.3, objval, places=1)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
