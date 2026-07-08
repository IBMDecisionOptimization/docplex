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
Tests the ConflictInterface.

No command line arguments are required.
"""
from collections import defaultdict
import unittest
import cplex
import os
from cplextestcase import CplexTestCase
from cplex.exceptions import (error_codes, CplexSolverError,
                              WrongNumberOfArgumentsError)


class ConflictInterfaceTests(CplexTestCase):
    """Basic tests on the ConflictInterface.

    See the test_explain test for more.
    """
    def testAllConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.all_constraints()
            self.isEmptyConstraintGroup(cons)

    def testUpperBoundConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.upper_bound_constraints()
            self.isEmptyConstraintGroup(cons)

    def testLowerBoundConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.lower_bound_constraints()
            self.isEmptyConstraintGroup(cons)

    def testLinearConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.linear_constraints()
            self.isEmptyConstraintGroup(cons)

    def testQuadraticConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.quadratic_constraints()
            self.isEmptyConstraintGroup(cons)

    def testIndicatorConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.indicator_constraints()
            self.isEmptyConstraintGroup(cons)

    def testSosConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.SOS_constraints()
            self.isEmptyConstraintGroup(cons)

    def testPwlConstraintsEmpty(self):
        with self._newCplex() as cpx:
            cons = cpx.conflict.pwl_constraints()
            self.isEmptyConstraintGroup(cons)

    def testRefineMipStartArgs(self):
        with self._newCplex() as cpx:
            with self.assertRaises(TypeError):
                # When we provide no arguments, we should get a
                # TypeError about missing required positional argument.
                cpx.conflict.refine_MIP_start()
            with self.assertRaises(CplexSolverError) as exc:
                # When we provide only the MIP_start argument, we should get
                # a CplexSolverError.
                cpx.conflict.refine_MIP_start(MIP_start=0)
            self.assertEqual(exc.exception.args[2], error_codes.CPXERR_NOT_MIP)
            with self.assertRaises(CplexSolverError) as exc:
                # If we pass in the conflict groups explicitly, then we
                # also expect a CplexSolverError.
                cpx.conflict.refine_MIP_start(
                    0, cpx.conflict.all_constraints())
            self.assertEqual(exc.exception.args[2], error_codes.CPXERR_NOT_MIP)

    def testRefineArgs(self):
        with self._newCplex() as cpx:
            cpx.conflict.refine()
            self.assertEqual(cpx.solution.get_status(),
                             cpx.solution.status.conflict_feasible)
            self.assertEqual(cpx.conflict.get_num_groups(), 0)

    def testRefineEmpty(self):
        with self._newCplex() as cpx:
            cpx.conflict.refine(cpx.conflict.all_constraints())
            try:
                cpx.conflict.get()
                self.fail()
            except CplexSolverError as cse:
                self.assertEqual(cse.args[2], error_codes.CPXERR_NO_CONFLICT)

    def testGetGroupsEmpty(self):
        with self._newCplex() as cpx:
            cpx.conflict.refine(cpx.conflict.all_constraints())
            self.assertEqual(cpx.solution.get_status(),
                             cpx.solution.status.conflict_feasible)
            with self.assertRaises(CplexSolverError) as exc:
                grps = cpx.conflict.get_groups()
            self.assertEqual(exc.exception.args[2],
                             error_codes.CPXERR_NO_CONFLICT)

    def testConflictFeasible(self):
        with self._newCplex() as cpx:
            cpx.read(self._getResource("tests/data/lpprog.lp"))
            cpx.conflict.refine()
            self.assertEqual(cpx.solution.status.conflict_feasible,
                             cpx.solution.get_status())
            with self.assertRaises(CplexSolverError) as exc:
                cpx.conflict.get_groups()
            self.assertEqual(exc.exception.args[2],
                             error_codes.CPXERR_NO_CONFLICT)
            self.assertEqual(cpx.conflict.get_num_groups(), 0)

    def testWriteEmpty(self):
        with self._newCplex() as cpx:
            try:
                with self._getTempFileName() as tmp:
                    cpx.conflict.write(tmp)
                self.fail()
            except CplexSolverError as cse:
                self.assertEqual(cse.args[2], error_codes.CPXERR_NO_CONFLICT)

    def isEmptyConstraintGroup(self, congrp):
        self.assertTrue(
            isinstance(congrp, cplex._internal._aux_functions._group))
        self.assertEqual([], congrp._gp)

    def testPwlConflict(self):
        with self._newCplex() as cpx:
            cpx.read(self._getResource("tests/data/pwlinf.lp"))
            cpx.conflict.refine(cpx.conflict.all_constraints())
            self.assertEqual(cpx.solution.status.conflict_minimal,
                             cpx.solution.get_status())
            self.assertGreater(
                cpx.solution.progress.get_num_conflict_passes(),
                0
            )
            confstatus = cpx.conflict.get()
            groups = cpx.conflict.get_groups()
            numgroups = cpx.conflict.get_num_groups()
            self.assertEqual(len(groups), numgroups)
            # extract group types
            conftypes = [grptype
                         for _, subgroup in groups
                         for grptype, _ in subgroup]
            # get counts for each group type
            grpdict = defaultdict(int)
            for grptype, grpstat in zip(conftypes, confstatus):
                if grpstat in (cpx.conflict.group_status.possible_member,
                               cpx.conflict.group_status.member):
                    grpdict[grptype] += 1
            # check the results
            constraint_type = cpx.conflict.constraint_type
            self.assertGreater(grpdict[constraint_type.lower_bound], 1)
            self.assertGreater(grpdict[constraint_type.linear], 1)
            self.assertGreater(grpdict[constraint_type.pwl], 1)

    def testMIPStartConflict(self):
        with self._newCplex() as cpx:
            cpx.read(self._getResource("examples/data/mexample.mps"))
            cpx.MIP_starts.add([["x1", "x2", "x3", "x4"],
                                [0.0, 0.0, 0.0, 0.0]],
                               cpx.MIP_starts.effort_level.auto,
                               "mst1")
            cpx.conflict.refine_MIP_start("mst1")
            self.assertEqual(cpx.solution.status.conflict_minimal,
                             cpx.solution.get_status())
            numpasses = cpx.solution.progress.get_num_conflict_passes()
            self.assertGreater(numpasses, 0)
            self.assertGreater(cpx.conflict.get_num_groups(), 0)

    def testInfeasible(self):
        with self._newCplex() as cpx:
            cpx.read(self._getResource("examples/data/infeasible.lp"))
            # When we call refine() with no arguments, conflict groups
            # are created automatically.
            cpx.conflict.refine()

            # In order to calculate the number of expected constraint
            # groups that were created using the method below, we need to
            # make sure that no binary variables exist.
            self.assertEqual(
                len([t for t in cpx.variables.get_types()
                     if t == 'B']),
                0
            )

            # Make sure that we only have linear constraints
            self.assertEqual(
                cpx.quadratic_constraints.get_num()
                + cpx.SOS.get_num()
                + cpx.pwl_constraints.get_num()
                + cpx.indicator_constraints.get_num(),
                0
            )
            self.assertGreater(cpx.linear_constraints.get_num(), 0)

            # Calculate expected number of constraint groups. This
            # requires knowledge from internal implementation.
            numgroups = (len([lb for lb in cpx.variables.get_lower_bounds()
                              if lb > -cplex.infinity])
                         + len([ub for ub in cpx.variables.get_upper_bounds()
                                if ub < cplex.infinity])
                         + cpx.linear_constraints.get_num())
            self.assertEqual(cpx.conflict.get_num_groups(), numgroups)
            self.assertEqual(numgroups, 8)

            # Test get_groups() functionality and exercise the different
            # ways that it can be called.
            # get_groups() with no args.
            groups = cpx.conflict.get_groups()

            # get_groups() with one args.
            for i in range(numgroups):
                g = cpx.conflict.get_groups(i)
                self.assertEqual(groups[i], g)

            # get_groups() with two args.
            for i in range(numgroups):
                for j in range(i, numgroups):
                    grps = cpx.conflict.get_groups(i, j)
                    for idx, g in zip(range(i, j), grps):
                        self.assertEqual(groups[idx], g)

            # get_groups() with a sequence.
            indices = [0, 3, 7]
            grps = cpx.conflict.get_groups(indices)
            for idx, g in zip(indices, grps):
                self.assertEqual(groups[idx], g)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
