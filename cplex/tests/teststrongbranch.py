# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Tests advanced.strongbranch()."""
from math import ceil, floor
from random import sample
import unittest
from cplextestcase import CplexTestCase
import cplex
from cplex.exceptions import CplexError, CplexSolverError, error_codes


class StrongBranchTests(CplexTestCase):

    def strongbranch_byhand(self, c, ilist):
        """ A function that does strong branching by hand.

        On a not too complicated model it should gave the same
        results as strongbranch.
        """
        penalties = []
        basis = c.solution.basis.get_basis()[0]
        lbs = c.variables.get_lower_bounds()
        ubs = c.variables.get_upper_bounds()
        x = c.solution.get_values()
        for i in ilist:
            # Down branch
            if basis[i] != c.solution.basis.status.basic:
                penalties.append((-1e20, -1e20))
                continue
            lb, ub = (lbs[i], ubs[i])
            c.variables.set_lower_bounds(i, ceil(x[i]))
            c.solve()
            if c.solution.get_status() == c.solution.status.infeasible:
                uppen = 1e75
            else:
                uppen = c.solution.get_objective_value()
            c.variables.set_lower_bounds(i, lb)

            # Up branch
            c.variables.set_upper_bounds(i, floor(x[i]))
            c.solve()
            if c.solution.get_status() == c.solution.status.infeasible:
                downpen = 1e75
            else:
                downpen = c.solution.get_objective_value()
            c.variables.set_upper_bounds(i, ub)
            penalties.append((downpen, uppen))
        return penalties

    def frac(self, v):
        return min(v-floor(v), ceil(v)-v)

    def construct_ilist(self, c, x, numvar=-1,
                        onlyfrac=True, fractionality=5e-3):
        self.assertEqual(len(x), c.variables.get_num())
        if numvar is None:
            numvars = len(x)
        if onlyfrac:
            ilist = [(i, v) for i, v in enumerate(x)
                     if self.frac(v) > fractionality]
        else:
            ilist = list(zip(range(len(x)), x))
        if numvars < len(x):
            ilist = sample(ilist, k=numvar)
        return sorted([i for i, v in ilist])

    def compareStrongBranchings(self, modelpath, numvar=None):
        """ Compare the results of CPXXstrongbranch with hand-made
        version

        Note that for results to be comparable we need to set a very large
        time limits. We should expect this to actually work for sufficiently
        simple models.
        """
        with self._newCplex() as c:
            c.read(modelpath)
            c.set_problem_type(c.problem_type.LP)
            c.solve()
            x = c.solution.get_values()
            ilist = self.construct_ilist(c, x, numvar, onlyfrac=True)
            res1 = self.strongbranch_byhand(c, ilist)
        with self._newCplex() as c:
            itlim = c.parameters.simplex.limits.iterations.get()
            c.read(modelpath)
            c.set_problem_type(c.problem_type.LP)
            c.solve()
            res2 = c.advanced.strong_branching(ilist, itlim)

        for r1, r2 in zip(res1, res2):
            self.assertAlmostEqual(r1[0], r2[0])
            self.assertAlmostEqual(r1[1], r2[1])

    def testp0033(self):
        self.compareStrongBranchings("../../data/p0033.mps")

    def testmyprobmip(self):
        self.compareStrongBranchings("../../data/myprobmip.lp")

    def testlocation_lin(self):
        self.compareStrongBranchings("../../data/location_lin.lp")

    def testSimpleIndices(self):
        c = self._newCplex()
        itlim = c.parameters.simplex.limits.iterations.get()
        c.read("../../data/example.mps")
        vars = list(range(c.variables.get_num()))
        result = c.advanced.strong_branching(vars, itlim)
        self.assertEqual(len(result), len(vars))

    def testSimpleNames(self):
        c = self._newCplex()
        itlim = c.parameters.simplex.limits.iterations.get()
        c.read("../../data/example.mps")
        vars = c.variables.get_names()
        result = c.advanced.strong_branching(vars, itlim)
        self.assertEqual(len(result), len(vars))

    def testBadIndexLow(self):
        c = self._newCplex()
        self.skipIfParamTesting(c)
        itlim = 1
        c.read("../../data/example.mps")
        vars = [-1]
        with self.assertRaises(CplexSolverError) as cm:
            c.advanced.strong_branching(vars, itlim)
        self.assertEqual(cm.exception.args[2],
                         error_codes.CPXERR_COL_INDEX_RANGE)

    def testBadIndexHigh(self):
        c = self._newCplex()
        self.skipIfParamTesting(c)
        itlim = 1
        c.read("../../data/example.mps")
        vars = [c.variables.get_num()]
        with self.assertRaises(CplexSolverError) as cm:
            c.advanced.strong_branching(vars, itlim)
        self.assertEqual(cm.exception.args[2],
                         error_codes.CPXERR_COL_INDEX_RANGE)

    def testNegativeItLim(self):
        c = self._newCplex()
        self.skipIfParamTesting(c)
        itlim = -1
        c.read("../../data/example.mps")
        vars = list(range(c.variables.get_num()))
        with self.assertRaises(CplexSolverError) as cm:
            c.advanced.strong_branching(vars, itlim)
        self.assertEqual(cm.exception.args[2],
                         error_codes.CPXERR_BAD_ARGUMENT)

    def testZeroItLim(self):
        c = self._newCplex()
        self.skipIfParamTesting(c)
        itlim = 0
        c.read("../../data/example.mps")
        vars = list(range(c.variables.get_num()))
        with self.assertRaises(CplexSolverError) as cm:
            c.advanced.strong_branching(vars, itlim)
        self.assertEqual(cm.exception.args[2],
                         error_codes.CPXERR_BAD_ARGUMENT)

    def testEmptyVarList(self):
        c = self._newCplex()
        self.skipIfParamTesting(c)
        itlim = c.parameters.simplex.limits.iterations.get()
        c.read("../../data/example.mps")
        vars = []
        result = c.advanced.strong_branching(vars, itlim)
        self.assertEqual([], result)

    def checkBadProblemType(self, model, expected_type, expected_error):
        c = self._newCplex()
        itlim = c.parameters.simplex.limits.iterations.get()
        c.read(model)
        self.assertEqual(c.get_problem_type(), expected_type)
        vars = list(range(c.variables.get_num()))
        with self.assertRaises(CplexSolverError) as cm:
            c.advanced.strong_branching(vars, itlim)
        self.assertEqual(cm.exception.args[2], expected_error)

    def testOnMIP(self):
        self.checkBadProblemType("../../data/caso8.mps",
                                 cplex.Cplex.problem_type.MILP,
                                 error_codes.CPXERR_NOT_FOR_MIP)

    def testOnQP(self):
        self.checkBadProblemType("../../data/qpex.lp",
                                 cplex.Cplex.problem_type.QP,
                                 error_codes.CPXERR_NOT_FOR_QP)

    def testOnQCP(self):
        self.checkBadProblemType("../../data/qcp.lp",
                                 cplex.Cplex.problem_type.QCP,
                                 error_codes.CPXERR_NOT_FOR_QCP)

    def testOnInfeasible(self):
        c = self._newCplex()
        self.skipIfParamTesting(c)
        itlim = c.parameters.simplex.limits.iterations.get()
        c.read("../../data/inflp.mps")
        vars = list(range(c.variables.get_num()))
        with self.assertRaises(CplexSolverError) as cm:
            c.advanced.strong_branching(vars, itlim)
        self.assertEqual(cm.exception.args[2],
                         error_codes.CPXERR_NEED_OPT_SOLN)


if __name__ == '__main__':
    unittest.main()
