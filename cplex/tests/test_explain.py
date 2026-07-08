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
Python implementation of the conflict refiner test.

See: testexplain.c, ilotestexplain.cpp, TestExplain.java.

No command line arguments are required.
"""
import os
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex.exceptions import (CplexError, CplexSolverError, error_codes,
                              WrongNumberOfArgumentsError)
from cplex import SparseTriple, SparsePair
from cplex.callbacks import IncumbentCallback

class MyCallback(IncumbentCallback):

    def __call__(self):
        print("new incumbent with objective value ",
              self.get_objective_value())

class EmptyCallback(IncumbentCallback):

    def __call__(self):
        pass

class ExplainTest(CplexTestCase):

    def testRefineNoArgs(self):
        cpx = self._newCplex()
        cpx.conflict.refine()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.conflict_feasible)

    def setupTest1(self):
        cpx = self._newCplex()
        cpx.set_problem_name('TestExplain')
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.variables.add(obj=[1, 0, 0],
                          lb=[6, -20, 8],
                          ub=[10, 10, 10],
                          names=['x', 'y', 'z'])
        cpx.linear_constraints.add(lin_expr=[[['x', 'y', 'z'], [1, 1, 1]],
                                             [['x'], [1]],
                                             [['y'], [1]],
                                             [['z'], [1]]],
                                   senses='LGGG',
                                   rhs=[15, 1, 7, 3],
                                   names=['rng', 'rng1', 'rng2', 'rng3'])
        cpx.solve()
        return cpx

    def test1_1(self):
        cpx = self.setupTest1()
        group_status = cpx.conflict.group_status

        # background
        refineargs = [cpx.conflict.linear_constraints(2., 'rng'),
                      cpx.conflict.linear_constraints(6., 'rng1'),
                      cpx.conflict.linear_constraints(7., 'rng2'),
                      cpx.conflict.linear_constraints(8., 'rng3'),
                      cpx.conflict.lower_bound_constraints(1., 'x'),
                      cpx.conflict.lower_bound_constraints(2., 'y'),
                      cpx.conflict.upper_bound_constraints(3., 'x'),
                      cpx.conflict.upper_bound_constraints(4., 'y'),
                      cpx.conflict.lower_bound_constraints(5., 'z'),
                      cpx.conflict.upper_bound_constraints(6., 'z')]
        expected = [group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member,
                    group_status.excluded,
                    group_status.excluded,
                    group_status.excluded,
                    group_status.excluded,
                    group_status.excluded,
                    group_status.possible_member,
                    group_status.excluded]
        self.conflict(cpx, refineargs, expected)

    def test1_2(self):
        cpx = self.setupTest1()
        group_status = cpx.conflict.group_status

        # 'c1' is background
        refineargs = [cpx.conflict.linear_constraints(6., 'rng1'),
                      cpx.conflict.linear_constraints(7., 'rng2'),
                      cpx.conflict.linear_constraints(8., 'rng3')]
        expected = [group_status.excluded,
                    group_status.possible_member,
                    group_status.excluded]
        self.conflict(cpx, refineargs, expected)

    def test1_3(self):
        cpx = self.setupTest1()
        group_status = cpx.conflict.group_status

        # using groups
        refineargs = [cpx.conflict.linear_constraints(2., 'rng'),
                      cpx.conflict.linear_constraints(6., 'rng1'),
                      (5., ((3, 'rng2'), (3, 'rng3')))]
        expected = [group_status.possible_member,
                    group_status.excluded,
                    group_status.possible_member]
        self.conflict(cpx, refineargs, expected)

    def test1_4(self):
        cpx = self.setupTest1()

        # repeated element
        refineargs = [cpx.conflict.linear_constraints(2., 'rng1'),
                      cpx.conflict.linear_constraints(6., 'rng1'),
                      cpx.conflict.linear_constraints(5., 'rng3')]
        try:
            self.conflict(cpx, refineargs, None)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_DUP_ENTRY)

    def test2(self):
        cpx = self._newCplex()
        cpx.variables.add(lb=[0], ub=[10], names=['x'])
        cpx.variables.add(lb=[0], ub=[10], names=['y'])
        cpx.variables.add(lb=[0], ub=[10], names=['z'])
        cpx.linear_constraints.add(
            lin_expr=[[['x', 'y', 'z'], [1, 1, 1]]],
            senses='G',
            rhs=[10],
            names=['rng'])
        cpx.quadratic_constraints.add(
            quad_expr=SparseTriple(ind1=['x'], ind2=['x'], val=[1]),
            sense='L',
            rhs=25,
            name='rng1')
        cpx.linear_constraints.add(
            lin_expr=[[['y'], [1]]],
            senses='L',
            rhs=[5],
            names=['rng2'])
        cpx.linear_constraints.add(
            lin_expr=[[['z'], [1]]],
            senses='L',
            rhs=[5],
            names=['rng3'])
        cpx.SOS.add(type=cpx.SOS.type.SOS1,
                    SOS=SparsePair(ind=['x', 'y', 'z'], val=[1, 2, 3]),
                    name='sos1')
        cpx.objective.set_sense(cpx.objective.sense.maximize)
        cpx.objective.set_linear('x', 1)

        cpx.register_callback(MyCallback)
        cpx.solve()
        self.assertEqual(cpx.solution.status.MIP_infeasible,
                         cpx.solution.get_status())
        cpx.register_callback(EmptyCallback)

        refineargs = [cpx.conflict.linear_constraints(1., 'rng'),
                      cpx.conflict.quadratic_constraints(1., 'rng1'),
                      cpx.conflict.linear_constraints(1., 'rng2'),
                      cpx.conflict.linear_constraints(1., 'rng3'),
                      cpx.conflict.SOS_constraints(1., 'sos1')]
        group_status = cpx.conflict.group_status
        expected = [group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member]
        self.conflict(cpx, refineargs, expected)

        # group containing a SOS
        refineargs = [cpx.conflict.linear_constraints(1., 'rng'),
                      cpx.conflict.quadratic_constraints(1., 'rng1'),
                      cpx.conflict.linear_constraints(1., 'rng2'),
                      (1., ((cpx.conflict.constraint_type.linear, 'rng3'),
                            (cpx.conflict.constraint_type.SOS, 'sos1')))]
        expected = [group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member]
        self.conflict(cpx, refineargs, expected)

        # group containing a SOS and quadratic constraint
        refineargs = [cpx.conflict.quadratic_constraints(1., 'rng1'),
                      cpx.conflict.linear_constraints(1., 'rng2'),
                      cpx.conflict.linear_constraints(1., 'rng3'),
                      (1., ((cpx.conflict.constraint_type.linear, 'rng'),
                            (cpx.conflict.constraint_type.SOS, 'sos1')))]
        expected = [group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member]
        self.conflict(cpx, refineargs, expected)

        # SOS in background
        refineargs = [cpx.conflict.linear_constraints(1., 'rng'),
                      cpx.conflict.quadratic_constraints(1., 'rng1'),
                      cpx.conflict.linear_constraints(1., 'rng2'),
                      cpx.conflict.linear_constraints(1., 'rng3')]
        expected = [group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member,
                    group_status.possible_member]
        self.conflict(cpx, refineargs, expected)

    def test3(self):
        cpx = self._newCplex()
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.variables.add(names=['x', 'y', 'z'], types=['B', 'B', 'B'])
        cpx.linear_constraints.add(
            lin_expr=[[['x', 'y', 'z'], [1, 1, 0]],
                      [['x', 'y', 'z'], [0, 1, 1]],
                      [['x', 'y', 'z'], [1, 0, 1]],
                      [['x', 'y', 'z'], [1, 1, 1]]],
            senses='LLLG',
            rhs=[1, 1, 1, 2],
            names=['rng1', 'rng2', 'rng3', 'rng4'])
        cpx.solve()

        self.assertEqual(cpx.parameters.read.datacheck.get(),
                         cpx.parameters.read.datacheck.values.warn)

        # tests for duplicates in groups
        refineargs = [cpx.conflict.linear_constraints(1., 'rng1'),
                      cpx.conflict.linear_constraints(1., 'rng2'),
                      (1., ((cpx.conflict.constraint_type.linear, 'rng1'),
                            (cpx.conflict.constraint_type.linear, 'rng3')))]
        try:
            self.conflict(cpx, refineargs, None)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_DUP_ENTRY)

    def test4(self):
        cpx = self._newCplex()
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        cpx.variables.add(names=['x', 'y', 'z'], types=['B', 'B', 'B'])
        cpx.linear_constraints.add(
            lin_expr=[[['x', 'y', 'z'], [1, 1, 0]],
                      [['x', 'y', 'z'], [0, 1, 1]],
                      [['x', 'y', 'z'], [1, 0, 1]]],
            senses='LLL',
            rhs=[1, 1, 1],
            names=['rng1', 'rng2', 'rng3'])
        cpx.solve()

        self.assertEqual(cpx.parameters.read.datacheck.get(),
                         cpx.parameters.read.datacheck.values.warn)

        # tests case when no conflicts exist
        refineargs = [cpx.conflict.linear_constraints(1., 'rng1'),
                      cpx.conflict.linear_constraints(1., 'rng2'),
                      cpx.conflict.linear_constraints(1., 'rng3')]
        try:
            self.conflict(cpx, refineargs, None)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_CONFLICT)

    def test5(self):
        """Cribbed from refine_MIP_start doctest."""
        cpx = self._newCplex()
        cpx.variables.add(obj=[1, 2], lb=[0, 0], ub=[0, 0],
                          types=[cpx.variables.type.binary,
                                 cpx.variables.type.binary])
        cpx.solve()

        cpx.linear_constraints.add(
            lin_expr=[[[0, 1], [1.0, 1.0]]], senses="E", rhs=[2.0])

        group_status = cpx.conflict.group_status
        expected_status = [group_status.excluded,
                           group_status.excluded,
                           group_status.excluded,
                           group_status.excluded,
                           group_status.member]
        expected_groups = [(1.0, ((2, 0),)), (1.0, ((2, 1),)),
                           (1.0, ((1, 0),)), (1.0, ((1, 1),)),
                           (1.0, ((3, 0),))]

        cpx.conflict.refine_MIP_start(0, cpx.conflict.all_constraints())
        self.assertEqual(cpx.conflict.get(), expected_status)
        self.assertEqual(cpx.conflict.get_groups(), expected_groups)

        # Now do it again with a hard-coded tuple
        refineargs = [cpx.conflict.upper_bound_constraints(1., 0),
                      cpx.conflict.upper_bound_constraints(1., 1),
                      cpx.conflict.lower_bound_constraints(1., 0),
                      cpx.conflict.lower_bound_constraints(1., 1),
                      (1.0, ((3, 0),))]
        cpx.conflict.refine_MIP_start(0, *refineargs);
        self.assertEqual(cpx.conflict.get(), expected_status)
        self.assertEqual(cpx.conflict.get_groups(), expected_groups)

    def test6(self):
        """Simple test with an indicator constraint."""
        cpx = self._newCplex()
        cpx.variables.add(obj=[1, 1], names=['x1', 'x2'],
                          lb=[1, 0], ub=[1, 0])
        cpx.indicator_constraints.add(indvar='x1',
                                      complemented=0,
                                      rhs=1.0,
                                      sense='G',
                                      lin_expr=SparsePair(ind=['x2'],
                                                          val=[2.0]),
                                      name='ind1')
        cpx.solve()

        # Run the test twice to make sure the names are deleted
        # (as with RTC-24001).
        for i in range(2):
            cpx.conflict.refine(cpx.conflict.all_constraints())
            self.assertEqual(cpx.conflict.get(), [-1, 3, 3, -1, 3])
            self.assertEqual(cpx.conflict.get_groups(),
                             [(1.0, ((2, 0),)), (1.0, ((2, 1),)),
                              (1.0, ((1, 0),)), (1.0, ((1, 1),)),
                              (1.0, ((6, 0),))])

    def conflict(self, cpx, refineargs, expected):
        # call the refiner
        cpx.parameters.conflict.algorithm.set(
            cpx.parameters.conflict.algorithm.values.fast)
        cpx.conflict.refine(*refineargs)
        # after calling refine, do a simple write test (for code coverage)
        conflictfile = 'conflict.clp'
        self._failSafeDelete(conflictfile)
        cpx.conflict.write(conflictfile)
        self.assertTrue(os.path.isfile(conflictfile))
        # make sure we got the expected results
        conf = cpx.conflict.get()
        self.assertEqual(conf, expected)
        # First, we must convert any names into indices. We use the
        # internal _separate_groups method to do this.
        separated_groups = cpx.conflict._separate_groups(refineargs)
        # Then, compose the group information back again.
        expectedgroups = cpx.conflict._compose_groups(*separated_groups)
        for i, group in enumerate(expectedgroups):
            self.assertEqual(group, cpx.conflict.get_groups(i))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
