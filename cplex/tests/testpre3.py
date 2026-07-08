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
Tests the Cplex.presolve.addrows() method.

This is the roughly the same as pretest3.c.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex import SparsePair
from cplex.exceptions import CplexError, CplexSolverError, error_codes


class Pre3Tests(CplexTestCase):

    def testAfiro(self):
        with self._newCplex() as cpx:
            # Set parameters.
            cpx.parameters.preprocessing.reduce.set(1)
            cpx.parameters.simplex.display.set(0)
            cpx.parameters.lpmethod.set(
                cpx.parameters.lpmethod.values.dual)
            cpx.parameters.advance.set(
                cpx.parameters.advance.values.none)

            # Read in the model.
            cpx.read("../../data/afiro.mps")

            # Solve the original model.
            cpx.solve()
            lpstat = cpx.solution.get_status()
            x = cpx.solution.get_values()
            objval = cpx.solution.get_objective_value()
            self.assertEqual(lpstat, cpx.solution.status.optimal)

            # Gather information and delete last block of constraints
            # (amount depends on delrat ratio).
            maxval = max([abs(j) for j in x])
            maxval *= 4.0
            self.assertLessEqual(maxval, cplex.infinity)

            delrat = 0.1
            numrows = cpx.linear_constraints.get_num()
            numcols = cpx.variables.get_num()
            delrows = int(delrat * numrows)
            if delrows == 0:
                delrows = 1
            begin = numrows - delrows
            end = numrows - 1
            rows = cpx.linear_constraints.get_rows(begin, end)
            nnz = cpx.linear_constraints.get_num_nonzeros()
            senses = cpx.linear_constraints.get_senses(begin, end)
            rhs = cpx.linear_constraints.get_rhs(begin, end)
            cpx.linear_constraints.delete(begin, end)

            # Change bounds depending on maxval (calculated above).
            lb = cpx.variables.get_lower_bounds()

            lu = ['?'] * numcols
            ind = [0] * numcols
            k = 0
            for j, bd in enumerate(lb):
                if bd < -maxval:
                    lb[k] = -maxval
                    lu[k] = 'L'
                    ind[k] = j
                    k += 1

            if k > 0:
                cpx.variables.set_lower_bounds(list(zip(ind[:k], lu[:k])))

            k = 0
            for j, bd in enumerate(lb):
                if bd > maxval:
                    lb[k] = maxval
                    lu[k] = 'U'
                    ind[k] = j
                    k += 1

            if k > 0:
                cpx.variables.set_upper_bounds(list(zip(ind[:k], lu[:k])))

            # Solve and check presolve status.
            cpx.solve()

            prestat = cpx.presolve.get_status()
            inpre = cpx.presolve.get_col_status()

            self.assertEqual(prestat, cpx.presolve.status.has_problem)

            # Change bounds depending on maxval (calculated above) and
            # whether variables remain in the presolve model.
            maxval /= 2.0

            lb = cpx.variables.get_lower_bounds()

            k = 0
            for j, bd in enumerate(lb):
                if bd < -maxval and inpre[j] >= 0:
                    lb[k] = -maxval
                    lu[k] = 'L'
                    ind[k] = j
                    k += 1

            if k > 0:
                cpx.variables.set_lower_bounds(list(zip(ind[:k], lu[:k])))

            k = 0
            for j, bd in enumerate(lb):
                if bd > maxval and inpre[j] >= 0:
                    lb[k] = maxval
                    lu[k] = 'U'
                    ind[k] = j
                    k += 1

            if k > 0:
                cpx.variables.set_upper_bounds(list(zip(ind[:k], lu[:k])))

            # Add the rows that we deleted above to the presolve model.
            cpx.presolve.add_rows(lin_expr=rows,
                                  senses=senses,
                                  rhs=rhs)

            prestat = cpx.presolve.get_status()
            self.assertEqual(prestat, cpx.presolve.status.has_problem)

            # Solve and check against original model.
            cpx.solve()

            lpstat = cpx.solution.get_status()
            objval1 = cpx.solution.get_objective_value()
            self.assertEqual(lpstat, cpx.solution.status.optimal)

            self.assertAlmostEqual(objval, objval1, 1e-5)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
