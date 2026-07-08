# --------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2007, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
""" We register a lazy constraint callback and an incumbent callback.
By definition the lazy constraint callback must be invoked before the
incumbent callback. In the lazy constraint callback we store the source
type as node data and in the incumbent callback we check that that is the
same as the incumbent callback claims.
"""
import sys
import unittest
import cplex
import os.path
from cplextestcase import CplexTestCase
from cplex.callbacks import LazyConstraintCallback
from cplex.callbacks import IncumbentCallback
from cplex.exceptions.errors import CplexSolverError
from cplex.exceptions import error_codes

HEURISTIC = "HEUR"
NODE = "NODE"
MIPSTART = "MIPSTART"


class CBHandle:

    def __init__(self):
        self.checked = False
        self.mstok = False
        self.tag = None


class MyLazyCallback(LazyConstraintCallback):

    def __init__(self, env):
        super().__init__(env)
        self.cbhandle = None
        self.testcase = None

    def __call__(self):
        try:
            source = self.get_solution_source()
            solsrc = IncumbentCallback.solution_source
            if source == solsrc.heuristic_solution:
                self.set_node_data(HEURISTIC)
            elif source == solsrc.node_solution:
                self.set_node_data(NODE)
            elif source == solsrc.mipstart_solution:
                if self.cbhandle.mstok:
                    # During MIP start processing there is no node context yet,
                    # so set_node_data() must fail.
                    with self.testcase.assertRaises(CplexSolverError) as cm:
                        self.set_node_data(MIPSTART)
                    self.testcase.assertEqual(cm.exception.args[2],
                                              error_codes.CPXERR_INDEX_RANGE)
                    self.cbhandle.tag = MIPSTART
                else:
                    self.testcase.fail('Unexpected mipstart_solution')
            else:
                self.testcase.fail(
                    'Invalid solution source in lazy constraint callback: {0}'
                    .format(source))
        except BaseException as err:
            print(err)
            raise


class MyIncumbentCallback(IncumbentCallback):

    def __init__(self, env):
        super().__init__(env)
        self.cbhandle = None
        self.testcase = None

    def __call__(self):
        try:
            source = self.get_solution_source()
            node_data = None
            if source != self.solution_source.mipstart_solution:
                node_data = self.get_node_data()
                self.testcase.assertIsNotNone(node_data)
            self.cbhandle.checked = True
            if source == self.solution_source.node_solution:
                self.testcase.assertEqual(node_data, NODE)
            elif source == self.solution_source.heuristic_solution:
                self.testcase.assertEqual(node_data, HEURISTIC)
            elif source == self.solution_source.mipstart_solution:
                if self.cbhandle.mstok:
                    # During MIP start processing there is no node context yet.
                    # So get_node_data() must fail
                    with self.testcase.assertRaises(CplexSolverError) as cm:
                        node_data = self.get_node_data()
                    self.testcase.assertEqual(cm.exception.args[2],
                                              error_codes.CPXERR_INDEX_RANGE)
                    node_data = self.cbhandle.tag
                    self.cbhandle.tag = None
                    self.testcase.assertEqual(node_data, MIPSTART)
                else:
                    self.testcase.fail('Unexpected mipstart_solution')
            else:
                self.testcase.fail(
                    'Unexpected solution source in incumbent callback: {0}'
                    .format(source))
        except BaseException as err:
            print(err)
            raise


class TestLazySourceCaso8(CplexTestCase):

    model_file = '../../data/caso8.mps'

    def testModelFile(self):
        self.assertTrue(self.model_file is not None)
        self.assertTrue(os.path.isfile(self.model_file))

    def testDefaults(self):
        cpx = self._newCplex()
        cpx.read(self.model_file)
        _, inc = self.registerCallbacks(cpx)
        cpx.solve()
        self.assertTrue(inc.cbhandle.checked)
        self.assertIsNone(inc.cbhandle.tag)

    def testMIPStart(self):
        cpx = self._newCplex()
        cpx.read(self.model_file)
        cpx.parameters.mip.limits.solutions.set(1)
        cpx.solve()
        startind = [x for x
                    in range(cpx.variables.get_num())
                    if cpx.variables.get_types(x) != 'C']
        startval = cpx.solution.get_values(startind)
        cpx.parameters.reset()
        cpx.read(self.model_file)
        cpx.MIP_starts.add([startind, startval],
                           cpx.MIP_starts.effort_level.auto)
        _, inc = self.registerCallbacks(cpx)
        inc.cbhandle.mstok = True
        cpx.solve()
        self.assertTrue(inc.cbhandle.checked)
        self.assertIsNone(inc.cbhandle.tag)

    def testMultiThreaded(self):
        cpx = self._newCplex()
        cpx.read(self.model_file)
        cpx.parameters.threads.set(cpx.get_num_cores())
        _, inc = self.registerCallbacks(cpx)
        cpx.solve()
        self.assertTrue(inc.cbhandle.checked)
        self.assertIsNone(inc.cbhandle.tag)

    def testOpportunistic(self):
        cpx = self._newCplex()
        cpx.read(self.model_file)
        cpx.parameters.threads.set(cpx.get_num_cores())
        cpx.parameters.parallel.set(
            cpx.parameters.parallel.values.opportunistic)
        _, inc = self.registerCallbacks(cpx)
        try:
            cpx.solve()
        except CplexSolverError as e:
            if e.args[2] == error_codes.CPXERR_SUBPROB_SOLVE:
                print("subproblem status: ",
                      cpx.solution.MIP.get_subproblem_status())
            raise
        self.assertTrue(inc.cbhandle.checked)
        self.assertIsNone(inc.cbhandle.tag)

    def registerCallbacks(self, cpx):
        cbhandle = CBHandle()
        self.assertFalse(cbhandle.checked)
        self.assertFalse(cbhandle.mstok)
        self.assertIsNone(cbhandle.tag)

        lazy = cpx.register_callback(MyLazyCallback)
        self.assertIsNone(lazy.cbhandle)
        lazy.cbhandle = cbhandle
        self.assertIsNone(lazy.testcase)
        lazy.testcase = self

        inc = cpx.register_callback(MyIncumbentCallback)
        self.assertIsNone(inc.cbhandle)
        inc.cbhandle = cbhandle
        self.assertIsNone(inc.testcase)
        inc.testcase = self

        return lazy, inc


class TestLazySourceAflow30a(TestLazySourceCaso8):

    model_file = '../../data/aflow30a.mps.gz'


def main():
    unittest.main()

if __name__ == '__main__':
    main()
