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
Tests docstring examples.

No command line arguments are required.
"""
import unittest
import doctest
import os
import cplex
from cplextestcase import CplexTestCase

NOT_EXPECTING_TESTS = ("Not expecting any docstring examples in this "
                       "module.  If that has changed then update this "
                       "test.")


class DocStringTests(CplexTestCase):
    # If you want to see the verbose doctest output, set this to True.
    verbose = False

    def testCplex(self):
        self.checkExamples(cplex)

    def testCplexCallbacks(self):
        self.checkNoExamples(cplex.callbacks)

    def testCplexExceptions(self):
        self.checkNoExamples(cplex.exceptions)

    def testCplexExceptionsErrorCodes(self):
        self.checkExamples(cplex.exceptions.error_codes)

    def testCplexInternal(self):
        self.checkNoExamples(cplex._internal)

    def testCplexInternalAuxFunctions(self):
        self.checkExamples(cplex._internal._aux_functions)

    def testCplexInternalConstants(self):
        self.checkNoExamples(cplex._internal._constants)

    def testCplexInternalListArrayUtils(self):
        self.checkNoExamples(cplex._internal._list_array_utils)

    def testCplexInternalMatrices(self):
        self.checkExamples(cplex._internal._matrices)

    def testCplexInternalOstream(self):
        self.checkNoExamples(cplex._internal._ostream)

    def testCplexInternalParameterClasses(self):
        self.checkExamples(cplex._internal._parameter_classes)

    def testCplexInternalParameterHierarchy(self):
        self.checkNoExamples(cplex._internal._parameter_hierarchy)

    def testCplexInternalParametersAuto(self):
        self.checkNoExamples(cplex._internal._parameters_auto)

    def testCplexInternalProcedural(self):
        self.checkNoExamples(cplex._internal._procedural)

    # NOTE: cplex._internal._pycplex_platform is not included in
    #       cplex._internal.__init__ so is not visible.

    def testCplexInternalPyCplex(self):
        self.checkNoExamples(cplex._internal._pycplex)

    def testCplexInternalSubInterfaces(self):
        self.checkExamples(cplex._internal._subinterfaces)

    def testCplexInternalAnno(self):
        self.checkExamples(cplex._internal._anno)

    def testCplexInternalPWL(self):
        self.checkExamples(cplex._internal._pwl)

    def testCplexInternalMultiObj(self):
        self.checkExamples(cplex._internal._multiobj)

    def testCplexInternalMultiObjSoln(self):
        self.checkExamples(cplex._internal._multiobjsoln)

    def testCplexParameterSet(self):
        self.checkExamples(cplex.paramset)

    # The lines above run all the doctest snippets in the modules passed
    # to the doctest.testmod function.  To run the doctest snippet for a
    # single method, use the following function call as a template.  Note
    # the second argument '{}' to the doctest.run_docstring_examples
    # function; it specifies that no global variables are defined for the
    # execution context of the test.
    #
    # doctest.run_docstring_examples(
    #     cplex._internal._subinterfaces.BasisInterface.get_basis_dual_norms,
    #     {})

    @classmethod
    def setUpClass(cls):
        """Called once before tests are run."""
        # In the docstring tests we read several different files.  Some of
        # these are renamed, some are not (those that have None below).
        cls.examples = {'afiro.mps':'lpex.mps',
                        'infeasible.lp':None,
                        'inflp.mps':None,
                        'example.mps':None,
                        'qp2qcp.lp':'qcp.lp',
                        'qpex.lp':'qp.lp',
                        'qpindef.lp':None,
                        'location_lin.lp':'ind.lp',
                        'p0033_qc1.lp':'miqcp.lp',
                        'unblp.lp':None,
                        'UFL_25_35_1.mps':None}

        # copy files for read methods
        for fkey, fval in cls.examples.items():
            if fval is None:
                fval = fkey
            os.system("cp ../../data/{0} ./{1}".format(fkey, fval))

    @classmethod
    def tearDownClass(cls):
        """Called once after all tests have run."""
        # clean up files copied for read methods
        for fkey, fval in cls.examples.items():
            if fval is None:
                fval = fkey
            cls._failSafeDelete("./{0}".format(fval))

        # remove files created by tests of write methods
        cls._failSafeDelete("./test_all.mst")
        cls._failSafeDelete("./test_one.mst")
        cls._failSafeDelete("./test_four.mst")
        cls._failSafeDelete("./ind.flt")

        # remove solution files created by tests of write solution methods
        cls._failSafeDelete("./lpex.sol")
        cls._failSafeDelete("./ind.sol")

        # remove annotation files created by docstring tests
        cls._failSafeDelete("./example.ann")
        cls._failSafeDelete("./UFL_25_35_1.ann")

    def checkExamples(self, module):
        """Tests that module has docstring examples and that they all pass."""
        (failure_count, test_count) = doctest.testmod(
            module, verbose=self.verbose)
        self.assertGreater(test_count, 0)
        self.assertEqual(failure_count, 0)

    def checkNoExamples(self, module):
        """Tests that module has no docstring examples."""
        (failure_count, test_count) = doctest.testmod(
            module, verbose=self.verbose)
        self.assertEqual(test_count, 0, NOT_EXPECTING_TESTS)
        self.assertEqual(failure_count, 0)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
