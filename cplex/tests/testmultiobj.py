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
Tests the multi-objective API.

No command line arguments are required.
"""
import unittest
import cplex
import testutil
from cplex.callbacks import Context
from cplex.exceptions import CplexSolverError
from cplex.exceptions import error_codes
from cplex._internal import ProblemType
from cplextestcase import CplexTestCase
from testutil import OutputProcessor


class AbortCallback():

    def __init__(self):
        self.num_calls = 0

    def invoke(self, context):
        self.num_calls += 1
        context.abort()


class RejectAllCallback():

    def __init__(self):
        self.num_calls = 0

    def invoke(self, context):
        self.num_calls += 1
        context.reject_candidate()


class MultiObjTests(CplexTestCase):

    def testGetNumEmpty(self):
        cpx = self._newCplex()
        # There is always one objective!
        self.assertEqual(cpx.multiobj.get_num(), 1)

    def testSetNumZero(self):
        cpx = self._newCplex()
        with self.assertRaises(CplexSolverError) as cm:
            cpx.multiobj.set_num(0)
        # FIXME: CPXERR_BAD_ARGUMENT is not very helpful....
        self.assertEqual(cm.exception.args[2],
                         error_codes.CPXERR_BAD_ARGUMENT)

    def testSetNumOne(self):
        cpx = self._newCplex()
        expected = [1.0] * 3
        indices = list(cpx.variables.add(obj=expected))
        cpx.multiobj.set_num(1)  # should be a no-op
        self.assertEqual(cpx.multiobj.get_num(), 1)
        actual = cpx.objective.get_linear()
        self.assertEqual(expected, actual)
        (actual, offset, weight, priority, abstol,
         reltol) = cpx.multiobj.get_definition(0)
        self.assertEqual(expected, actual)
        self.assertEqual(0.0, offset)
        self.assertEqual(1.0, weight)
        self.assertEqual(0, priority)
        self.assertEqual(0.0, abstol)
        self.assertEqual(0.0, reltol)

    def checkDefaults(self, cpx, objidx, expected_obj=None):
        # Allow for a non-default objective.
        if expected_obj is None:
            expected_obj = [0.0] * cpx.variables.get_num()
        expected_abstol, expected_reltol = 0.0, 0.0
        (obj, offset, weight, priority,
         abstol, reltol) = cpx.multiobj.get_definition(objidx)
        self.assertEqual(expected_obj, obj)
        self.assertEqual(0.0, offset)
        self.assertEqual(1.0, weight)
        self.assertEqual(0, priority)
        self.assertEqual(expected_abstol, abstol)
        self.assertEqual(expected_reltol, reltol)

    def testSetNumTwo(self):
        cpx = self._newCplex()
        expected = [1.0] * 3
        indices = list(cpx.variables.add(obj=expected))
        cpx.multiobj.set_num(2)  # add one new objective
        self.assertEqual(cpx.multiobj.get_num(), 2)
        actual = cpx.objective.get_linear()
        self.assertEqual(expected, actual)
        actual = cpx.multiobj.get_linear(0)
        self.assertEqual(expected, actual)
        self.checkDefaults(cpx, 0, expected)
        self.checkDefaults(cpx, 1)

    def testSetNameTrad(self):
        cpx = self._newCplex()
        name = "foo"
        cpx.objective.set_name(name)
        self.assertEqual(name, cpx.objective.get_name())
        self.assertEqual(name, cpx.multiobj.get_names(0))

    def checkSetName(self, objidx):
        cpx = self._newCplex()
        cpx.multiobj.set_num(objidx + 1)
        expected = "foo"
        cpx.multiobj.set_name(objidx, name=expected)
        self.assertEqual(expected, cpx.multiobj.get_names(objidx))
        if objidx == 0:
            self.assertEqual(expected, cpx.objective.get_name())

    def testSetName(self):
        for i in range(3):
            self.checkSetName(i)

    def testSetNameMore(self):
        cpx = self._newCplex()
        before = 5
        after = before * 2
        cpx.multiobj.set_num(before)
        names = [str(i) for i in range(before)]
        for i, name in enumerate(names):
            cpx.multiobj.set_name(i, name)
        self.assertEqual(names, cpx.multiobj.get_names())
        cpx.multiobj.set_num(after)
        try:
            cpxnames = cpx.multiobj.get_names()
        except cplex.exceptions.CplexSolverError as e:
            self.assertEqual(e.args[2], error_codes.CPXERR_NO_NAMES)
        cpx.multiobj.set_num(before)
        self.assertEqual(names, cpx.multiobj.get_names())

    def testSetLinearTrad(self):
        cpx = self._newCplex()
        indices = list(cpx.variables.add(lb=[0.0] * 3))
        expected = [1.0] * 3
        cpx.objective.set_linear(list(zip(indices, expected)))
        self.assertEqual(expected, cpx.objective.get_linear())
        (actual, _, _, _, _, _) = cpx.multiobj.get_definition(0)
        self.assertEqual(expected, actual)

    def checkSetUp(self, objidx, use_names):
        """Generic setup for "check" methods."""
        cpx = self._newCplex()
        cpx.multiobj.set_num(objidx + 1)
        if use_names:
            obj_id = "obj{0}".format(objidx)
            cpx.multiobj.set_name(objidx, obj_id)
            self.assertEqual(obj_id, cpx.multiobj.get_names(objidx))
        else:
            obj_id = objidx
        return cpx, obj_id

    def checkSetLinear(self, objidx, use_names):
        cpx, obj_id = self.checkSetUp(objidx, use_names)
        indices = list(cpx.variables.add(lb=[0.0] * 3))
        expected = [1.0] * 3
        cpx.multiobj.set_linear(obj_id, list(zip(indices, expected)))
        (actual, _, _, _, _, _) = cpx.multiobj.get_definition(obj_id)
        self.assertEqual(expected, actual)
        actual = cpx.multiobj.get_linear(obj_id)
        self.assertEqual(expected, actual)
        if objidx == 0:
            self.assertEqual(expected, cpx.objective.get_linear())

    def testSetLinear(self):
        for i in range(3):
            self.checkSetLinear(i, False)

    def testSetLinearByName(self):
        for i in range(3):
            self.checkSetLinear(i, True)

    def testSetLinearSingle(self):
        cpx = self._newCplex()
        testutil.create_lpex1(cpx)
        cpx.multiobj.set_num(2)
        cpx.multiobj.set_linear(1, 0, 1)
        self.assertEqual([1.0, 0.0, 0.0], cpx.multiobj.get_linear(1))

    def testClearLinear(self):
        cpx = self._newCplex()
        indices = list(cpx.variables.add(obj=[1.0] * 3))
        self.assertEqual([1.0] * 3, cpx.objective.get_linear())
        cpx.multiobj.set_definition(0)
        self.assertEqual([0.0] * 3, cpx.objective.get_linear())

    def testSetOffsetTrad(self):
        cpx = self._newCplex()
        expected = 1.0
        cpx.objective.set_offset(expected)
        (_, offset, _, _, _, _) = cpx.multiobj.get_definition(0)
        self.assertEqual(expected, offset)
        offset = cpx.multiobj.get_offset(0)
        self.assertEqual(expected, offset)
        offset = cpx.objective.get_offset()
        self.assertEqual(expected, offset)

    def checkSetOffset(self, objidx, use_names):
        cpx, obj_id = self.checkSetUp(objidx, use_names)
        expected = 1.0
        cpx.multiobj.set_offset(obj_id, expected)
        (_, offset, _, _, _, _) = cpx.multiobj.get_definition(obj_id)
        self.assertEqual(expected, offset)
        offset = cpx.multiobj.get_offset(obj_id)
        self.assertEqual(expected, offset)
        if objidx == 0:
            self.assertEqual(expected, cpx.objective.get_offset())

    def testSetOffset(self):
        for i in range(3):
            self.checkSetOffset(i, False)

    def testSetOffsetByName(self):
        for i in range(3):
            self.checkSetOffset(i, True)

    def testSetSense(self):
        cpx = self._newCplex()
        cpx.multiobj.set_sense(cpx.multiobj.sense.maximize)
        self.assertEqual(cpx.multiobj.sense.maximize,
                         cpx.multiobj.get_sense())
        self.assertEqual(cpx.objective.sense.maximize,
                         cpx.objective.get_sense())

    def checkSetWeight(self, objidx, use_names):
        cpx, obj_id = self.checkSetUp(objidx, use_names)
        expected = 2.0
        cpx.multiobj.set_weight(obj_id, expected)
        (_, _, weight, _, _, _) = cpx.multiobj.get_definition(obj_id)
        self.assertEqual(expected, weight)
        weight = cpx.multiobj.get_weight(obj_id)
        self.assertEqual(expected, weight)

    def testSetWeight(self):
        for i in range(3):
            self.checkSetWeight(i, False)

    def testSetWeightByName(self):
        for i in range(3):
            self.checkSetWeight(i, True)

    def checkSetPriority(self, objidx, use_names):
        cpx, obj_id = self.checkSetUp(objidx, use_names)
        expected = 2
        cpx.multiobj.set_priority(obj_id, expected)
        (_, _, _, priority, _, _) = cpx.multiobj.get_definition(obj_id)
        self.assertEqual(expected, priority)
        priority = cpx.multiobj.get_priority(obj_id)
        self.assertEqual(expected, priority)

    def testSetPriority(self):
        for i in range(3):
            self.checkSetPriority(i, False)

    def testSetPriorityByName(self):
        for i in range(3):
            self.checkSetPriority(i, True)

    def checkSetAbsTol(self, objidx, use_names):
        cpx, obj_id = self.checkSetUp(objidx, use_names)
        expected = 0.1
        cpx.multiobj.set_abstol(obj_id, expected)
        (_, _, _, _, abstol, _) = cpx.multiobj.get_definition(obj_id)
        self.assertEqual(expected, abstol)
        abstol = cpx.multiobj.get_abstol(obj_id)
        self.assertEqual(expected, abstol)

    def testSetAbsTol(self):
        for i in range(3):
            self.checkSetAbsTol(i, False)

    def testSetAbsTolByName(self):
        for i in range(3):
            self.checkSetAbsTol(i, True)

    def checkSetRelTol(self, objidx, use_names):
        cpx, obj_id = self.checkSetUp(objidx, use_names)
        expected = 0.1
        cpx.multiobj.set_reltol(obj_id, expected)
        (_, _, _, _, _, reltol) = cpx.multiobj.get_definition(obj_id)
        self.assertEqual(expected, reltol)
        reltol = cpx.multiobj.get_reltol(obj_id)
        self.assertEqual(expected, reltol)

    def testSetRelTol(self):
        for i in range(3):
            self.checkSetRelTol(i, False)

    def testSetRelTolByName(self):
        for i in range(3):
            self.checkSetRelTol(i, True)

    # FIXME: Add tests with negative weights (for different objective
    #        senses).

    def create_lpex1_multiobj(self):
        cpx = self._newCplex()
        testutil.create_lpex1(cpx)
        cpx.multiobj.set_num(2)
        cpx.multiobj.set_linear(
            1,
            [(idx, 1.0) for idx in range(cpx.variables.get_num())]
        )
        cpx.multiobj.set_priority(0, 1)
        cpx.multiobj.set_priority(1, 2)
        return cpx

    def testSimpleMultiObjOpt(self):
        cpx = self.create_lpex1_multiobj()
        cpx.solve()
        self.assertEqual(
            202.5,
            cpx.solution.multiobj.get_objective_value(0)
        )
        self.assertEqual(
            100.0,
            cpx.solution.multiobj.get_objective_value(1)
        )

    def testMultiObjInfeasibleLP(self):
        cpx = self.create_lpex1_multiobj()
        cpx.linear_constraints.add([[["x1"], [1.0]]],
                                   rhs=[50.0],
                                   senses="G",
                                   names=["c_infeasible"])
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.multiobj_infeasible)

    def testMultiObjUnboundedLP(self):
        cpx = self.create_lpex1_multiobj()
        cpx.variables.set_upper_bounds("x1", cplex.infinity)
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.multiobj_unbounded)

    def testMultiObjNonOptimalMIP(self):
        cpx = self._newCplex()
        cpx.read("../../data/dietmultiobj.lp")
        ps1 = cpx.create_parameter_set()
        ps2 = cpx.create_parameter_set()
        # Remark: If we set the dettimelimit to 0.0, we get
        #         status multiobj_stopped instead of
        #         multiobj_non_optimal but this would be fine.
        #         See discussion in RTC-38604.
        ps2.add(cpx.parameters.dettimelimit, 0.1)
        cpx.solve([ps1, ps2])
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.multiobj_non_optimal)

    def testMultiobjInfOrUnbdMIP(self):
        cpx = self._newCplex()
        cpx.read("../../data/infmip.lp")
        self.assertEqual(cpx.multiobj.get_num(), 1)
        # Make a copy of the objective to make a multiobjective model.
        cpx.multiobj.set_num(2)
        cpx.multiobj.set_linear(
            1,
            [(idx, val) for idx, val in enumerate(cpx.objective.get_linear())]
        )
        cpx.multiobj.set_priority(0, 2)
        cpx.multiobj.set_priority(0, 1)
        # Solve and check for expected status.
        cpx.solve()
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.multiobj_inforunbd)

    def testMultiObjLimitLP(self):
        cpx = self.create_lpex1_multiobj()
        self.assertEqual(cpx.multiobj.get_num(), 2)
        # Set the hidden CPX_PARAM_MULTIOBJLIMIT parameter.
        CPX_PARAM_MULTIOBJLIMIT = 1601
        cpx.parameters._set(CPX_PARAM_MULTIOBJLIMIT, 1)
        cpx.solve()
        self.assertEqual(
            cpx.solution.get_status(),
            cpx.solution.status.multiobj_stopped
        )
        self.assertEqual(cpx.solution.multiobj.get_num_solves(), 1)

    def testMultiObjLimitMIP(self):
        cpx = self._newCplex()
        cpx.read("../../data/dietmultiobj.lp")
        self.assertEqual(cpx.multiobj.get_num(), 2)
        # Set the hidden CPX_PARAM_MULTIOBJLIMIT parameter.
        CPX_PARAM_MULTIOBJLIMIT = 1601
        cpx.parameters._set(CPX_PARAM_MULTIOBJLIMIT, 1)
        cpx.solve()
        self.assertEqual(
            cpx.solution.get_status(),
            cpx.solution.status.multiobj_stopped
        )
        self.assertEqual(cpx.solution.multiobj.get_num_solves(), 1)

    def testMultiObjAbortFromCB(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.read("../../data/dietmultiobj.lp")
        cb = AbortCallback()
        cpx.set_callback(cb, Context.id.candidate)
        cpx.solve()
        self.assertGreater(cb.num_calls, 0)
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.multiobj_stopped)

    def testMultiObjRejectAllFromCB(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.parameters.dettimelimit.set(25.0)
        cpx.read("../../data/dietmultiobj.lp")
        cb = RejectAllCallback()
        cpx.set_callback(cb, Context.id.candidate)
        cpx.solve()
        self.assertGreater(cb.num_calls, 0)
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.multiobj_stopped)

    def testSimpleMultiObjOptByPriority(self):
        cpx = self.create_lpex1_multiobj()
        cpx.solve()
        self.assertEqual(
            202.5,
            cpx.solution.multiobj.get_objval_by_priority(1)
        )
        self.assertEqual(
            100.0,
            cpx.solution.multiobj.get_objval_by_priority(2)
        )

    def testSimpleMultiObjOptAdvanced(self):
        cpx = self.create_lpex1_multiobj()
        ps1 = cpx.create_parameter_set()
        ps2 = cpx.create_parameter_set()
        cpx.solve(paramsets=[ps1, ps2])
        self.assertEqual(
            202.5,
            cpx.solution.multiobj.get_objective_value(0)
        )
        self.assertEqual(
            100.0,
            cpx.solution.multiobj.get_objective_value(1)
        )

    def testSimpleMultiObjOptWithNone(self):
        """Test solve(paramsets) where some items in the list are None."""
        cpx = self.create_lpex1_multiobj()
        ps1 = cpx.create_parameter_set()
        ps2 = None  # This is supported.
        cpx.solve(paramsets=[ps1, ps2])

    def checkMultiObjDisplay(self, cpx, display_value, expected_matches,
                             test_mip=True):
        """Basic sanity checks on multi-objective display output.

        cpx - a new Cplex object
        display_value - one of cpx.parameters.multiobjective.display.values
        expected_matches - a three-tuple containing expected number of
                           matches (see below)
        test_mip - True for testing MIP, False for testing LP
        """
        if test_mip:
            cpx.read("../../data/dietmultiobj.lp")
            regex_list = ("Multi-objective solve log",
                          "Parallel mode",
                          r"^\nFinished optimization #.*with priority [0-9]+\.")
        else:
            cpx = self.create_lpex1_multiobj()
            regex_list = ("Multi-objective solve log",
                          "Iteration log",
                          "Iterations =",
                          "Index Priority")
        cpx.parameters.multiobjective.display.set(display_value)
        self.assertEqual(len(regex_list), len(expected_matches))
        outproc = OutputProcessor(regex_list)
        self._setAllStreams(cpx, outproc)
        cpx.solve()
        self.assertEqual(
            cpx.solution.status.multiobj_optimal,
            cpx.solution.get_status()
        )
        self.assertEqual(cpx.solution.multiobj.get_num_solves(), 2)
        for item, expected in zip(outproc.regex_list, expected_matches):
            self.assertEqual(
                item.num_matches, expected,
                "Matched '{0}' {1} times, expected {2}.".format(
                    item.regex_string, item.num_matches, expected))

    def testMultiObjMIPDisplayNone(self):
        cpx = self._newCplex()
        expected_matches = (0, 0, 0)
        self.checkMultiObjDisplay(
            cpx,
            cpx.parameters.multiobjective.display.values.none,
            expected_matches)

    def testMultiObjMIPDisplayNormal(self):
        cpx = self._newCplex()
        expected_matches = (1, 0, 0)
        self.checkMultiObjDisplay(
            cpx,
            cpx.parameters.multiobjective.display.values.normal,
            expected_matches)

    def testMultiObjMIPDisplayDetailed(self):
        cpx = self._newCplex()
        expected_matches = (1, 2, 2)
        self.checkMultiObjDisplay(
            cpx,
            cpx.parameters.multiobjective.display.values.detailed,
            expected_matches)

    def testMultiObjLPDisplayNone(self):
        cpx = self._newCplex()
        expected_matches = (0, 0, 0, 0)
        self.checkMultiObjDisplay(
            cpx,
            cpx.parameters.multiobjective.display.values.none,
            expected_matches,
            False)

    def testMultiObjLPDisplayNormal(self):
        cpx = self._newCplex()
        expected_matches = (1, 0, 0, 1)
        self.checkMultiObjDisplay(
            cpx,
            cpx.parameters.multiobjective.display.values.normal,
            expected_matches,
            False)

    def testMultiObjLPDisplayDetailed(self):
        cpx = self._newCplex()
        expected_matches = (1, 2, 2, 0)
        self.checkMultiObjDisplay(
            cpx,
            cpx.parameters.multiobjective.display.values.detailed,
            expected_matches,
            False)

    def testDetailedOutputNoBlending(self):
        """Check that blending info is not printed in some cases."""
        cpx = self._newCplex()
        cpx.parameters.multiobjective.display.set(
            cpx.parameters.multiobjective.display.values.detailed)
        cpx.read("../../data/dietmultiobj.lp")
        outproc = OutputProcessor(["Starting optimization #1 with priority 2",
                                   "Finished optimization #1 with priority 2",
                                   "Starting optimization #2 with priority 1",
                                   "Finished optimization #2 with priority 1"])
        self._setAllStreams(cpx, outproc)
        cpx.solve()
        self.assertEqual(cpx.solution.multiobj.get_num_solves(), 2)
        int_info = cpx.solution.multiobj.int_info
        self.assertEqual(cpx.solution.multiobj.get_info(0, int_info.blend), 1)
        self.assertEqual(cpx.solution.multiobj.get_info(1, int_info.blend), 1)
        for item in outproc.regex_list:
            self.assertEqual(item.num_matches, 1)

    def testDetailedOutputWithBlending(self):
        """Check that extra blending info is printed when appropriate."""
        cpx = self._newCplex()
        cpx.parameters.multiobjective.display.set(
            cpx.parameters.multiobjective.display.values.detailed)
        cpx.read("../../data/multiobj.lp")
        outproc = OutputProcessor(["Starting optimization #1 blending 2 objectives with priority 2",
                                   "Finished optimization #1 of 2 blended objectives with priority 2",
                                   "Starting optimization #2 with priority 1",
                                   "Finished optimization #2 with priority 1"])
        self._setAllStreams(cpx, outproc)
        cpx.solve()
        self.assertEqual(cpx.solution.multiobj.get_num_solves(), 2)
        int_info = cpx.solution.multiobj.int_info
        self.assertEqual(cpx.solution.multiobj.get_info(0, int_info.blend), 2)
        self.assertEqual(cpx.solution.multiobj.get_info(1, int_info.blend), 1)
        for item in outproc.regex_list:
            self.assertEqual(item.num_matches, 1)

    def checkNormalBlendingOutput(self, modelfile, modeltype):
        """Check that a "Blend" column is printed with normal display level."""
        cpx = self._newCplex()
        # Normal is the default:
        self.assertEqual(cpx.parameters.multiobjective.display.get(),
                         cpx.parameters.multiobjective.display.values.normal)
        cpx.read(modelfile)
        # Check expected model type:
        self.assertEqual(cpx.get_problem_type(), modeltype)
        outproc = OutputProcessor(["Index +Priority +Blend"])
        self._setAllStreams(cpx, outproc)
        cpx.solve()
        self.assertEqual(cpx.solution.multiobj.get_num_solves(), 2)
        for item in outproc.regex_list:
            self.assertEqual(item.num_matches, 1)

    def testNormalBlendingOutputLP(self):
        self.checkNormalBlendingOutput("../../data/multiobj.lp",
                                       ProblemType.LP)

    def testNormalBlendingOutputLP(self):
        self.checkNormalBlendingOutput("../../data/dietmultiobj.lp",
                                       ProblemType.MILP)

    def testSubProblemParameterOutput(self):
        """We should see global parameters and local parameters if we set
        multi-objective display to detailed.
        """
        cpx = self._newCplex()
        cpx.read("../../data/dietmultiobj.lp")
        cpx.parameters.multiobjective.display.set(
            cpx.parameters.multiobjective.display.values.detailed)
        cpx.parameters.dettimelimit.set(15000)  # global limit
        ps1 = cpx.create_parameter_set()
        ps1.add(cpx.parameters.dettimelimit, 11000)  # local limit
        ps2 = cpx.create_parameter_set()
        ps2.add(cpx.parameters.dettimelimit, 9000)  # local limit
        outproc = OutputProcessor(["CPXPARAM_DetTimeLimit *15000",
                                   "CPXPARAM_DetTimeLimit *11000",
                                   "CPXPARAM_DetTimeLimit *9000"])
        self._setAllStreams(cpx, outproc)
        cpx.solve(paramsets=[ps1, ps2])
        self.assertEqual(
            cpx.solution.status.multiobj_optimal,
            cpx.solution.get_status()
        )
        self.assertEqual(cpx.solution.multiobj.get_num_solves(), 2)
        for item in outproc.regex_list:
            self.assertEqual(item.num_matches, 1)

    def testSingleObjWithParamSets(self):
        cpx = self._newCplex()
        ps = cpx.create_parameter_set()
        with self.assertRaises(ValueError) as err:
            cpx.solve(paramsets=[ps])
        self.assertIn("paramsets argument can only be specified for a"
                      " multi-objective model", str(err.exception))

    def testNotEnoughParamSets(self):
        """Test solve(paramsets) with fewer items in list than the number
        of priorities. We expect an exception.
        """
        cpx = self.create_lpex1_multiobj()
        ps1 = cpx.create_parameter_set()
        with self.assertRaises(ValueError) as err:
            cpx.solve(paramsets=[ps1])
        self.assertIn("if specified, len(paramsets)", str(err.exception))

    def testTooManyParamSets(self):
        """Test solve(paramsets) with more items in list than the number
        of priorities. We expect an exception.
        """
        cpx = self.create_lpex1_multiobj()
        ps1 = [cpx.create_parameter_set()
               for _ in range(cpx.multiobj.get_num() + 1)]
        with self.assertRaises(ValueError) as err:
            cpx.solve(paramsets=[ps1])
        self.assertIn("if specified, len(paramsets)", str(err.exception))

    def testMoreObjectivesThanPriorities(self):
        cpx = self._newCplex()
        cpx.read("../../data/multiobj.lp");
        nobjs = cpx.multiobj.get_num()
        self.assertEqual(nobjs, 3)
        nprios = len(set([cpx.multiobj.get_priority(i)
                          for i in range(cpx.multiobj.get_num())]))
        self.assertEqual(nprios, 2)
        self.assertGreater(nobjs, nprios)
        paramsets = [cpx.create_parameter_set() for _ in range(nprios)]
        _ = [p.read("../../data/dettime.prm") for p in paramsets]
        cpx.solve(paramsets)
        self.assertEqual(cpx.solution.get_status(),
                         cpx.solution.status.multiobj_optimal)

    def testErrorOnQP(self):
        cpx = self._newCplex()
        cpx.read('../../data/qpex.lp')
        cpx.multiobj.set_num(2)
        with self.assertRaises(CplexSolverError) as err:
            cpx.solve()
        self.assertEqual (err.exception.args[2],
                          error_codes.CPXERR_NOT_FOR_QP)

    def testErrorOnQCP(self):
        cpx = self._newCplex()
        cpx.read('../../data/qcp.lp')
        cpx.multiobj.set_num(2)
        with self.assertRaises(CplexSolverError) as err:
            cpx.solve()
        self.assertEqual (err.exception.args[2],
                          error_codes.CPXERR_NOT_FOR_QCP)

    def testErrorOnPopulate(self):
        cpx = self._newCplex()
        cpx.read('../../data/p0033.mps')
        cpx.multiobj.set_num(2)
        with self.assertRaises(CplexSolverError) as err:
            cpx.populate_solution_pool()
        self.assertEqual (err.exception.args[2],
                          error_codes.CPXERR_NOT_FOR_MULTIOBJ)

    def testErrorOnBenders(self):
        cpx = self._newCplex()
        cpx.read('../../data/UFL_25_35_1.mps')
        cpx.multiobj.set_num(2)
        cpx.parameters.benders.strategy.set(3)
        with self.assertRaises(CplexSolverError) as err:
            cpx.solve()
        self.assertEqual (err.exception.args[2],
                          error_codes.CPXERR_NOT_FOR_BENDERS)
        cpx.read_annotations('../../data/UFL_25_35_1.ann')
        cpx.parameters.benders.strategy.set(0)
        with self.assertRaises(CplexSolverError) as err:
            cpx.solve()
        self.assertEqual (err.exception.args[2],
                          error_codes.CPXERR_NOT_FOR_BENDERS)

    def testCleanup(self):
        cpx = self._newCplex()
        cpx.variables.add(lb=[0.0] * 3)
        cpx.multiobj.set_num(3)
        # Set each multiobj to have at least one small NZ value.
        for idx, coef in enumerate([1e-3, 1e-4, 1e-5]):
            cpx.multiobj.set_linear(idx, idx, coef)
        cpx.cleanup(1e-2)
        # After cleanup we expect the objectives to be zeroed out.
        for idx in range(cpx.multiobj.get_num()):
            actual = cpx.multiobj.get_linear(idx)
            self.assertEqual([0.0] * 3, actual)

    def testGetDefinitionByName(self):
        cpx = self._newCplex()
        varnames = ["x1", "x2", "x3"]
        obj0 = [1.0, 1.0, 1.0]
        expected_objdef0 = [obj0, 0.0, 1.0, 0, 0.0, 0.0]
        objname0 = "obj1"  # default name

        # First, set up the first objective using defaults for all
        # multiobj attributes.
        varind = list(cpx.variables.add(obj=obj0, names=varnames))

        # Second, set up a second objective with non-default multiobj
        # attributes.
        cpx.multiobj.set_num(2)
        obj1 = [2.0, 2.0, 2.0]
        expected_objdef1 = [obj1, 1.0, -1.0, 1, 2e-6, 3e-4]
        (_, offset, weight, priority, abstol, reltol) = expected_objdef1
        objname1 = "obj2"
        cpx.multiobj.set_definition(1,
                                    obj=[varind, obj1],
                                    offset=offset,
                                    weight=weight,
                                    priority=priority,
                                    abstol=abstol,
                                    reltol=reltol,
                                    name=objname1)

        # Check expected definitions.
        for idx, (objname, objdef) in enumerate(
                ((objname0, expected_objdef0),
                 (objname1, expected_objdef1))
        ):
            self.assertEqual(objname, cpx.multiobj.get_names(idx))
            # Expected definition
            (expobj, expoffset, expweight, exppriority, expabstol,
             expreltol) = objdef
            # Query by name and use optional begin and end arguments.
            (actobj, actoffset, actweight, actpriority, actabstol,
             actreltol) = cpx.multiobj.get_definition(objname,
                                                      begin=varnames[0],
                                                      end=varnames[-1])
            self.assertEqual(actobj, expobj)
            self.assertEqual(actoffset, expoffset)
            self.assertEqual(actweight, expweight)
            self.assertEqual(actpriority, exppriority)
            self.assertEqual(actabstol, expabstol)
            self.assertEqual(actreltol, expreltol)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
