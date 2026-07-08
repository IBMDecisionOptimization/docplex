# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Tests advanced.basicpresolve()."""
import unittest
from cplextestcase import CplexTestCase, enumerate_reversed
from cplex.exceptions import CplexError, error_codes


class BasicPresolveTests(CplexTestCase):

    def strengthen(self, cpx):
        redlb, redub, rstat = cpx.advanced.basic_presolve()
        self.assertEqual(len(redlb), cpx.variables.get_num())
        self.assertEqual(len(redub), cpx.variables.get_num())
        self.assertEqual(len(rstat), cpx.linear_constraints.get_num())
        # FIXME: It's annoying that we can't query types if the model is
        #        not a MIP. As is, we currently get a CPXERR_NOT_MIP.
        #        Should we just return ['C'] * numcols?

        # FIXME: It's annoying that we can't pass in an empty list to
        #        functions like set_types. Should we make it a no-op?

        # FIXME: Not all of this code actually gets exercised (i.e., the
        #        models don't necessarily trigger any of it).

        # Demote semi-continuous variables to continuous.
        semicont = []
        try:
            semicont.extend([(idx, cpx.variables.type.continuous)
                             for idx, ctype
                             in enumerate(cpx.variables.get_types())
                             if ctype == cpx.variables.type.semi_continuous
                             and redlb[idx] > 0.0])
        except CplexError as err:
            self.assertEqual(err.args[2], error_codes.CPXERR_NOT_MIP)
        if len(semicont) > 0:
            cpx.variables.set_types(semicont)

        # Demote semi-integer variables to integer.
        semiint = []
        try:
            semiint.extend([(idx, cpx.variables.type.integer)
                            for idx, ctype
                            in enumerate(cpx.variables.get_types())
                            if ctype == cpx.variables.type.semi_integer
                            and redlb[idx] > 0.0])
        except CplexError as err:
            self.assertEqual(err.args[2], error_codes.CPXERR_NOT_MIP)
        if len(semiint) > 0:
            cpx.variables.set_types(semiint)

        # Strengthen bounds.
        cpx.variables.set_lower_bounds([(idx, lb)
                                        for idx, lb
                                        in enumerate(redlb)])
        cpx.variables.set_upper_bounds([(idx, ub)
                                        for idx, ub
                                        in enumerate(redub)])

        # Delete redundant rows.
        cpx.linear_constraints.delete([idx
                                       for idx, rst
                                       in enumerate_reversed(rstat)
                                       if rst != 0])

    def compareOriginalVsStrengthened(self, modelpath):
        with self._newCplex() as cpx:
            cpx.read(modelpath)
            cpx.solve()
            objval1 = cpx.solution.get_objective_value()
        with self._newCplex() as cpx:
            cpx.read(modelpath)
            self.strengthen(cpx)
            cpx.solve()
            objval2 = cpx.solution.get_objective_value()
        self.assertAlmostEqual(objval1, objval2)

    def testlpprog(self):
        self.compareOriginalVsStrengthened(self._getResource("tests/data/lpprog.lp"))

    def testmyprobmip(self):
        self.compareOriginalVsStrengthened("../../data/myprobmip.lp")

    def testpopulate(self):
        self.compareOriginalVsStrengthened("../../data/populate.lp")

    def testlocation_lin(self):
        self.compareOriginalVsStrengthened("../../data/location_lin.lp")

    def testqpex(self):
        self.compareOriginalVsStrengthened("../../data/qpex.lp")

    def testqcp(self):
        self.compareOriginalVsStrengthened("../../data/qcp.lp")


if __name__ == '__main__':
    unittest.main()
